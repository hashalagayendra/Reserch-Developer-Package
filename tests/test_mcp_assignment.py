"""Verify planner isolation and per-agent MCP/API tool binding without model calls."""
import importlib
import unittest
from unittest.mock import patch

from langchain_core.tools import StructuredTool
from rich.table import Table
from research.workflows.main_workflow.schemas import NonExecutableSubAgentConfig, OrchestratorOutput

main_module = importlib.import_module("research.workflows.main_workflow.main_workflow")
builder_module = importlib.import_module(
    "research.workflows.main_workflow.utility.build_layers_with_attaching_executable_agents"
)
single_module = importlib.import_module(
    "research.workflows.main_workflow.utility.create_single_agent"
)
logger_module = importlib.import_module("research.workflows.main_workflow.workflow_logger")


class MCPAssignmentTests(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def remote(value: str) -> str:
            """Echo a value from the remote service."""
            self.calls.append(value)
            return value

        self.remote = StructuredTool.from_function(remote)

    def config(self, **changes):
        data = dict(name="worker", system_prompt="Research", task="Find facts",
                    model_name="deepseek/deepseek-v4-flash", tools=[], mcp_tool_names=[])
        data.update(changes)
        return data

    def test_planner_only_receives_catalog_and_discovery(self):
        discovery = StructuredTool.from_function(lambda: [], name="tool_discovery_workflow",
                                                  description="Discover API tools")
        with patch.object(main_module, "ChatOpenRouter"), \
             patch.object(main_module, "create_react_agent") as create, \
             patch.object(main_module, "transform_tool_discovery_workflow_as_tool", return_value=discovery):
            workflow = main_module.MainWorkFlow(all_mcp_tools=[self.remote])
        bound = create.call_args.kwargs["tools"]
        self.assertEqual([t.name for t in bound], ["tool_discovery_workflow", "extract_mcp_tools"])
        catalog = workflow.mcp_discovery_tool.invoke({})
        self.assertEqual(catalog[0]["name"], "remote")
        self.assertIn("value", catalog[0]["input_schema"]["properties"])
        self.assertEqual(self.calls, [])

    def test_builder_combines_tools_only_for_assigned_agent(self):
        api = dict(tool_title="lookup", description="Lookup facts", api_url="https://example.test")
        layers = [dict(layer_number=1, sub_agents=[
            NonExecutableSubAgentConfig(**self.config(tools=[api], mcp_tool_names=["remote"])),
            self.config(name="reporter"),
        ])]
        with patch.object(single_module, "ChatOpenRouter"), \
             patch.object(single_module, "create_agent") as create, \
             patch.object(builder_module.WorkflowLogger, "log_executable_layers"):
            result = builder_module.build_layers_with_attaching_executable_agents(
                layers, all_mcp_tools=[self.remote], response_schema=NonExecutableSubAgentConfig
            )
        worker, reporter = result.layers[0].sub_agents
        self.assertEqual([t.name for t in worker.tools], ["lookup", "remote"])
        self.assertIs(worker.tools[1], self.remote)
        self.assertEqual(reporter.tools, [])
        self.assertIsNone(create.call_args_list[0].kwargs["response_format"])
        self.assertIs(create.call_args_list[1].kwargs["response_format"], NonExecutableSubAgentConfig)
        self.assertEqual(worker.tools[1].invoke({"value": "fact"}), "fact")
        self.assertEqual(self.calls, ["fact"])

    def test_unknown_assignment_is_rejected(self):
        layers = [dict(layer_number=1, sub_agents=[self.config(mcp_tool_names=["missing"])])]
        with self.assertRaisesRegex(ValueError, "Unknown MCP tool 'missing'"):
            builder_module.build_layers_with_attaching_executable_agents(layers, all_mcp_tools=[self.remote])

    def test_duplicate_names_across_sources_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate tool names"):
            single_module.create_single_agent(
                "test-model", "Research", "Find facts", tools=[self.remote], assigned_mcp_tools=[self.remote]
            )

    def test_existing_config_defaults_to_no_mcp_tools(self):
        data = self.config()
        del data["mcp_tool_names"]
        self.assertEqual(NonExecutableSubAgentConfig(**data).mcp_tool_names, [])

    def test_plan_table_displays_mcp_assignments_for_dict_and_model(self):
        api = dict(tool_title="lookup", description="Lookup facts", api_url="https://example.test")
        plan = dict(layers=[dict(layer_number=1, sub_agents=[
            self.config(name="student_marks_analyzer", mcp_tool_names=["get_students"]),
            self.config(name="project_info_retriever", tools=[api], mcp_tool_names=["get_project_info"]),
            self.config(name="report_synthesizer"),
        ])])
        for result in [plan, OrchestratorOutput(**plan)]:
            with self.subTest(result_type=type(result).__name__):
                with patch.object(logger_module.console, "print") as output:
                    logger_module.WorkflowLogger.log_orchestrator_result("Report", result)
                table = next(
                    call.args[0] for call in output.call_args_list
                    if call.args and isinstance(call.args[0], Table)
                )
                self.assertEqual(list(table.columns[-1].cells), [
                    "MCP: get_students", "lookup, MCP: get_project_info", "[dim]None[/dim]",
                ])


if __name__ == "__main__":
    unittest.main()
