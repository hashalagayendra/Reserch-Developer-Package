"""Opt-in checks against the actual sibling MCP HTTP server; no tool mocks."""

import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest

from research import Research
from research.config import settings
from research.mcp_tools import discover_mcp_tools


ENABLED = any(os.getenv(name) == "1" for name in (
    "MCP_INTEGRATION_TEST", "MCP_LIVE_AGENT_TEST",
))


@unittest.skipUnless(ENABLED, "Set MCP_INTEGRATION_TEST=1 to start the real MCP server")
class MCPIntegrationTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        server_dir = Path(__file__).resolve().parents[2] / "MCP_Server"
        server_file = server_dir / "server.py"
        if not server_file.is_file():
            raise RuntimeError("Place MCP_Server beside Reserch-Developer-Package")
        local_python = server_dir / (
            ".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python"
        )
        python = os.getenv("MCP_SERVER_PYTHON") or (
            str(local_python) if local_python.is_file() else sys.executable
        )
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        cls.url = f"http://127.0.0.1:{port}/mcp"
        log = tempfile.TemporaryFile(mode="w+b")
        cls.addClassCleanup(log.close)
        process = subprocess.Popen(
            [python, str(server_file), "--transport", "http", "--port", str(port)],
            cwd=server_dir, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

        def stop_server():
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)

        cls.addClassCleanup(stop_server)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                log.seek(0)
                raise RuntimeError("MCP server exited:\n" + log.read().decode(errors="replace"))
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    return
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("MCP server did not start within 30 seconds")

    async def test_real_http_discovery_and_invocation(self):
        tools = {tool.name: tool for tool in await discover_mcp_tools(self.url)}
        self.assertEqual(set(tools), {"greet", "get_students", "get_project_info"})
        self.assertIn("name", tools["greet"].args)
        result = await asyncio.wait_for(
            tools["greet"].ainvoke({"name": "integration-check"}), timeout=20,
        )
        self.assertIn("Hello, integration-check! Welcome to MCP.", str(result))

    async def test_research_initialization_loads_real_tools(self):
        research = Research()
        # Construction only: this key is never sent to a model provider.
        research.configure(openRouterAPIKey="unused-initialization-key", mcp_server_url=self.url)
        await research.initialize()
        self.assertEqual(
            {tool.name for tool in research.all_mcp_tools},
            {"greet", "get_students", "get_project_info"},
        )
        self.assertEqual(research.main_workflow.all_mcp_tools, research.all_mcp_tools)

    @unittest.skipUnless(os.getenv("MCP_LIVE_AGENT_TEST") == "1",
                         "Set MCP_LIVE_AGENT_TEST=1 for paid real-model execution")
    async def test_real_research_agent_uses_remote_result(self):
        self.assertTrue(settings.OPENROUTER_API_KEY, "Configure OPENROUTER_API_KEY first")
        model = os.getenv("MCP_TEST_MODEL") or settings.MAIN_WORKFLOW_MODEL
        research = Research()
        research.configure(
            mcp_server_url=self.url,
            mainWorkflowModel=model,
            toolDiscoveryModel=model,
        )
        result = await asyncio.wait_for(research.run(
            "Use extract_mcp_tools to inspect the MCP catalog. Create a minimal workflow "
            f"using model {model} for every sub-agent. Assign get_students via "
            "mcp_tool_names and actually call it. Do not use API discovery or other "
            "external tools. Find the student with the highest marks from the tool "
            "result. Your final response must include that student's name and marks."
        ), timeout=180)
        execution = result["execution_result"]
        observations = []
        for layer in execution["execution_history"]:
            for agent in layer["sub_agent_results"]:
                self.assertEqual(agent["status"], "success", agent.get("error"))
                messages = agent["messages"]
                calls = {
                    call["id"]
                    for message in messages if message.get("type") == "ai"
                    for call in message.get("tool_calls", [])
                    if call["name"] == "get_students"
                }
                observations.extend(
                    message for message in messages
                    if message.get("type") == "tool"
                    and message.get("tool_call_id") in calls
                    and message.get("status", "success") == "success"
                )
        self.assertTrue(observations, "No successful get_students tool-call/result pair recorded")
        self.assertIn("Amal", json.dumps(observations))
        self.assertIn("82", json.dumps(observations))
        final = execution["final_response"]
        self.assertIn("Amal", final)
        self.assertIn("82", final)


if __name__ == "__main__":
    unittest.main()
