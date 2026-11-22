"""Exercise the TypeSafe integration and failure handling without paid API calls."""
import importlib
import json
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx2
from langchain_typesafe import Choice, ClassifierResponse, TypeSafeClassifier

from research.workflows.main_workflow.utility.error_correction_agent.error_correction_sub_agent import ErrorCorrectionSubAgent
from research.workflows.main_workflow.utility.layer_execution_engine import run_single_agent

correction_module = importlib.import_module(ErrorCorrectionSubAgent.__module__)


class ErrorClassificationTests(unittest.IsolatedAsyncioTestCase):
    def make_handler(self, transport_handler):
        transport = httpx2.MockTransport(transport_handler)
        client = httpx2.Client(transport=transport)
        async_client = httpx2.AsyncClient(transport=transport)
        self.addCleanup(client.close)
        self.addAsyncCleanup(async_client.aclose)
        classifier = TypeSafeClassifier(
            api_key="test-key", base_url="https://openrouter.ai/api",
            model="typesafe/jev-1.14", client=client, async_client=async_client,
        )
        with patch.object(correction_module, "ChatOpenRouter"), \
             patch.object(correction_module, "create_agent"), \
             patch.object(correction_module, "TypeSafeClassifier", return_value=classifier) as initialize:
            handler = ErrorCorrectionSubAgent(api_key="test-key")
        initialize.assert_called_once_with(
            api_key="test-key", base_url="https://openrouter.ai/api", model="typesafe/jev-1.14",
        )
        return handler

    @staticmethod
    def response(category="infrastructure", probabilities=None, confidence=0.95):
        return httpx2.Response(200, json={
            "model": "typesafe/jev-1.14",
            "answers": {"error_category": {
                "type": "choice", "choice": category,
                "probabilities": probabilities if probabilities is not None else {
                    "model": 0.02, "infrastructure": 0.95,
                    "tool": 0.02, "structured_output": 0.01,
                }, "confidence": confidence,
            }},
            "usage": {"input_tokens": 10, "output_tokens": 0},
        })

    async def test_systemone_request_and_typed_response(self):
        captured = []

        def handle(request):
            captured.append(json.loads(request.content))
            self.assertEqual(str(request.url), "https://openrouter.ai/api/v1/systemone")
            self.assertEqual(request.headers["Authorization"], "Bearer test-key")
            return self.response()

        handler = self.make_handler(handle)
        result = await handler.classify_error({
            "error": TimeoutError("Request timed out"), "agent_name": "worker", "task": "Find facts",
        })
        self.assertEqual(result.model_dump(), {
            "category": "infrastructure", "confidence": 0.95,
            "probabilities": {"model": 0.02, "infrastructure": 0.95,
                              "tool": 0.02, "structured_output": 0.01},
        })
        self.assertEqual(captured[0]["model"], "typesafe/jev-1.14")
        self.assertEqual(captured[0]["state"]["error_type"], "TimeoutError")
        self.assertEqual(
            set(captured[0]["questions"]["error_category"]["criteria"]),
            {"model", "infrastructure", "tool", "structured_output"},
        )

    async def test_start_fix_invokes_typesafe_classifier(self):
        handler = self.make_handler(lambda request: self.response())
        handler.error_classifier = Mock(ainvoke=AsyncMock(return_value=
            ClassifierResponse.model_validate(self.response().json())
        ))
        error = TimeoutError("original failure")
        result = await handler.start_fix(
            {"name": "worker", "task": "Find facts"}, error=error,
        )
        self.assertEqual(result, {
            "category": "infrastructure", "confidence": 0.95,
            "probabilities": {"model": 0.02, "infrastructure": 0.95,
                              "tool": 0.02, "structured_output": 0.01},
        })
        handler.error_classifier.ainvoke.assert_awaited_once()
        payload = handler.error_classifier.ainvoke.call_args.args[0]
        self.assertEqual(payload["state"]["error"], "original failure")
        self.assertEqual(payload["state"]["error_type"], "TimeoutError")
        self.assertIsInstance(payload["questions"]["error_category"], Choice)

    async def test_classifier_failure_preserves_original_error_and_checkpoint(self):
        handler = self.make_handler(lambda request: httpx2.Response(503))
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

    async def test_highest_probability_wins_for_each_category(self):
        for winner in ("model", "infrastructure", "tool", "structured_output"):
            with self.subTest(winner=winner):
                scores = {"model": 0.1, "infrastructure": 0.1,
                          "tool": 0.1, "structured_output": 0.1}
                scores[winner] = 0.7
                # Deliberately differ from the returned choice and global confidence.
                handler = self.make_handler(lambda request: self.response(
                    category="model", probabilities=scores, confidence=0.4,
                ))
                result = await handler.classify_error({"error": RuntimeError("failure")})
                self.assertEqual(result.category, winner)
                self.assertEqual(result.confidence, 0.7)
                self.assertEqual(result.probabilities, scores)

    async def test_invalid_probability_distributions_are_rejected(self):
        for scores in (
            {"model": 0.8, "infrastructure": 0.1, "tool": 0.1},
            {"invented": 0.7, "infrastructure": 0.1, "tool": 0.1, "structured_output": 0.1},
            {"model": 1.1, "infrastructure": 0.0, "tool": 0.0, "structured_output": -0.1},
        ):
            with self.subTest(scores=scores):
                handler = self.make_handler(lambda request: self.response(probabilities=scores))
                with self.assertRaises(ValueError):
                    await handler.classify_error({"error": RuntimeError("failure")})


if __name__ == "__main__":
    unittest.main()
