"""Exercise the Decisions contract and failure handling without paid API calls."""
import importlib
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from research.workflows.main_workflow.utility.error_correction_agent.error_correction_sub_agent import ErrorClassifier, ErrorClassification
from research.workflows.main_workflow.utility.error_correction_agent.error_correction_sub_agent import ErrorCorrectionSubAgent
from research.workflows.main_workflow.utility.layer_execution_engine import run_single_agent

classifier_module = importlib.import_module(ErrorClassifier.__module__)
correction_module = importlib.import_module(ErrorCorrectionSubAgent.__module__)


class ErrorClassificationTests(unittest.IsolatedAsyncioTestCase):
    async def test_decisions_request_and_typed_response(self):
        captured = []

        def handle(request):
            import json
            captured.append(json.loads(request.content))
            self.assertEqual(request.url.path, "/api/alpha/decisions")
            self.assertEqual(request.headers["Authorization"], "Bearer test-key")
            return httpx.Response(200, json={"answers": {"error_category": {
                "type": "choice", "choice": "timeout", "confidence": 0.95,
            }}})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        with patch.object(classifier_module.httpx, "AsyncClient", return_value=client):
            result = await ErrorClassifier("test-key").ainvoke({
                "error": TimeoutError("Request timed out"), "agent_name": "worker", "task": "Find facts",
            }, config={"tags": ["error-classification"]})
        self.assertEqual(result.model_dump(), {"category": "timeout", "confidence": 0.95})
        self.assertEqual(captured[0]["model"], "typesafe/jev-1.14")
        self.assertEqual(captured[0]["state"]["error_type"], "TimeoutError")
        self.assertIn("unknown", captured[0]["questions"]["error_category"]["criteria"])

    async def test_start_fix_invokes_langchain_classifier(self):
        with patch.object(correction_module, "ChatOpenRouter"), \
             patch.object(correction_module, "create_agent"):
            handler = ErrorCorrectionSubAgent(api_key="test-key")
        handler.error_classifier.ainvoke = AsyncMock(
            return_value=ErrorClassification(category="timeout", confidence=0.95),
        )
        error = TimeoutError("original failure")
        result = await handler.start_fix(
            {"name": "worker", "task": "Find facts"}, error=error,
        )
        self.assertEqual(result, {"category": "timeout", "confidence": 0.95})
        handler.error_classifier.ainvoke.assert_awaited_once_with({
            "error": error, "agent_name": "worker", "task": "Find facts", "model_name": None,
        })

    async def test_classifier_failure_preserves_original_error_and_checkpoint(self):
        with patch.object(correction_module, "ChatOpenRouter"), \
             patch.object(correction_module, "create_agent"):
            handler = ErrorCorrectionSubAgent(api_key="test-key")
        handler.error_classifier.ainvoke = AsyncMock(side_effect=ValueError("bad response"))
        executor = AsyncMock()
        executor.ainvoke.side_effect = RuntimeError("original failure")
        checkpoints = []
        result = await run_single_agent(
            {"name": "worker", "task": "Find facts", "executor": executor},
            checkpoints=checkpoints, layer_number=1, error_correction_agent=handler,
        )
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error, "original failure")
        self.assertIn("original failure", checkpoints[0].agents[0].status)

    async def test_invalid_category_is_rejected(self):
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request:
            httpx.Response(200, json={"answers": {"error_category": {
                "choice": "invented", "confidence": 0.9,
            }}})))
        with patch.object(classifier_module.httpx, "AsyncClient", return_value=client):
            with self.assertRaises(ValueError):
                await ErrorClassifier("test-key").ainvoke({
                    "error": RuntimeError("failure"), "agent_name": "worker", "task": "Find facts",
                })


if __name__ == "__main__":
    unittest.main()
