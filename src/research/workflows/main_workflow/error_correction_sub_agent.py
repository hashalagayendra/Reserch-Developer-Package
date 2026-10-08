# """
# Error Correction Sub-Agent for Main Workflow.

# This agent intervenes when a sub-agent encounters an error or fails during execution.
# It analyzes the failure context, diagnoses the root cause, formulates a recovery plan,
# and attempts to correct the prompt/tool calls to recover successfully.
# """

# import json
# from typing import Optional, List, Dict, Any, Union
# from pydantic import BaseModel, Field, ConfigDict
# from langchain_core.prompts import PromptTemplate
# from langchain_openrouter import ChatOpenRouter
# from research.config import settings

 
# class ErrorDiagnosis(BaseModel):
#     """Structured diagnosis of the failure produced by the Error Correction Agent."""
#     error_type: str = Field(description="Category of the error (e.g., 'tool_argument_error', 'api_failure', 'parsing_error', 'context_missing')")
#     root_cause: str = Field(description="Detailed explanation of why the failure occurred")
#     correction_strategy: str = Field(description="Actionable strategy to resolve or bypass the error")
#     suggested_instructions: str = Field(description="Specific guidance or corrected input to provide to the agent for the retry")


# class ErrorCorrectionSubAgent:
#     """
#     Sub-agent responsible for analyzing execution errors, repairing agent states,
#     and retrying failed tasks with corrective guidance.
#     """

#     def __init__(
#         self,
#         api_key: Optional[str] = None,
#         model_name: Optional[str] = None,
#         temperature: float = 0.2,
#         max_retries: int = 2
#     ):
#         self.api_key = api_key or settings.OPENROUTER_API_KEY
#         self.model_name = model_name or settings.MAIN_WORKFLOW_MODEL
#         self.temperature = temperature
#         self.max_retries = max_retries

#         self.llm = ChatOpenRouter(
#             model=self.model_name,
#             temperature=self.temperature,
#             api_key=self.api_key
#         )

#         self.diagnosis_prompt = PromptTemplate(
#             template=(
#                 "You are an expert AI Diagnostic and Error Correction Agent.\n"
#                 "A sub-agent encountered an error during task execution. Analyze the failure and provide structured diagnosis.\n\n"
#                 "================== SUB-AGENT CONTEXT ==================\n"
#                 "Agent Name: {agent_name}\n"
#                 "Assigned Task: {task}\n"
#                 "Model: {model_name}\n"
#                 "Available Tools: {tools_available}\n\n"
#                 "================== FAILURE DETAILS ====================\n"
#                 "Error Encountered: {error_message}\n"
#                 "Context from Previous Layers: {accumulated_context}\n\n"
#                 "Diagnose why the sub-agent failed and provide specific, concrete correction instructions."
#             ),
#             input_variables=[
#                 "agent_name",
#                 "task",
#                 "model_name",
#                 "tools_available",
#                 "error_message",
#                 "accumulated_context"
#             ]
#         )

#     async def diagnose_error(
#         self,
#         agent_name: str,
#         task: str,
#         model_name: str,
#         error_message: str,
#         tools_available: List[str],
#         accumulated_context: Optional[str] = None
#     ) -> ErrorDiagnosis:
#         """
#         Diagnoses why a sub-agent failed and generates actionable correction instructions.
#         """
#         formatted_prompt = self.diagnosis_prompt.format(
#             agent_name=agent_name,
#             task=task,
#             model_name=model_name,
#             tools_available=", ".join(tools_available) if tools_available else "None",
#             error_message=str(error_message),
#             accumulated_context=accumulated_context or "None"
#         )

#         structured_llm = self.llm.with_structured_output(ErrorDiagnosis)
#         diagnosis: ErrorDiagnosis = await structured_llm.ainvoke(formatted_prompt)
#         return diagnosis

#     async def attempt_recovery(
#         self,
#         agent: Any,
#         error_message: str,
#         accumulated_context: Optional[str] = None,
#         retry_count: int = 1
#     ) -> Dict[str, Any]:
#         """
#         Diagnoses the error and executes a guided retry on the agent's executor.

#         Args:
#             agent: The ExecutableSubAgent instance that failed.
#             error_message: The exception message that caused the failure.
#             accumulated_context: Context from previous workflow layers.
#             retry_count: Current retry attempt number.

#         Returns:
#             Dict containing recovery status, diagnosis, and new response or error.
#         """
#         agent_name = getattr(agent, "name", "SubAgent")
#         agent_task = getattr(agent, "task", "")
#         agent_model = getattr(agent, "model_name", self.model_name)
#         executor = getattr(agent, "executor", None)
#         tools = getattr(agent, "tools", [])
#         tool_names = [t.name if hasattr(t, "name") else str(t) for t in tools] if tools else []

#         print(f"\n   🛠️  [ERROR CORRECTION SUB-AGENT ACTIVATED]")
#         print(f"      ├─ Target Agent: {agent_name} (Retry #{retry_count})")
#         print(f"      └─ Analyzing Error: {error_message}")

#         try:
#             # 1. Run diagnostic reasoning
#             diagnosis = await self.diagnose_error(
#                 agent_name=agent_name,
#                 task=agent_task,
#                 model_name=agent_model,
#                 error_message=error_message,
#                 tools_available=tool_names,
#                 accumulated_context=accumulated_context
#             )

#             print(f"      ├─ 🔍 Diagnosis Type: {diagnosis.error_type}")
#             print(f"      ├─ 📌 Root Cause: {diagnosis.root_cause}")
#             print(f"      └─ 💡 Strategy: {diagnosis.correction_strategy}")

#             if not executor:
#                 return {
#                     "recovered": False,
#                     "diagnosis": diagnosis.model_dump(),
#                     "response": f"Cannot retry: executor is missing for agent '{agent_name}'.",
#                     "error": error_message
#                 }

#             # 2. Build corrected user message injecting diagnostic guidance
#             corrected_user_message = (
#                 f"Assigned Task: {agent_task}\n\n"
#                 f"Previous Attempt Failed With: {error_message}\n\n"
#                 f"Diagnostic Guidance & Correction Instructions:\n"
#                 f"- Error Category: {diagnosis.error_type}\n"
#                 f"- Root Cause: {diagnosis.root_cause}\n"
#                 f"- Suggested Correction: {diagnosis.suggested_instructions}\n\n"
#             )

#             if accumulated_context:
#                 corrected_user_message += (
#                     f"Context & Results from Previous Layer(s):\n{accumulated_context}\n\n"
#                 )

#             corrected_user_message += "Please execute the task again, applying the diagnostic guidance above."

#             print(f"   🔄 [RE-EXECUTING SUB-AGENT WITH CORRECTIVE GUIDANCE]...")
#             res = await executor.ainvoke({"messages": [("user", corrected_user_message)]})

#             messages = res.get("messages", []) if isinstance(res, dict) else []
#             final_content = messages[-1].content if messages else ""

#             print(f"   🎉 [RECOVERY SUCCESSFUL]: {agent_name} recovered on retry #{retry_count}!")

#             return {
#                 "recovered": True,
#                 "diagnosis": diagnosis.model_dump(),
#                 "response": str(final_content),
#                 "messages": messages,
#                 "error": None
#             }

#         except Exception as retry_err:
#             print(f"   ⚠️  [RECOVERY ATTEMPT FAILED]: {str(retry_err)}")
#             return {
#                 "recovered": False,
#                 "diagnosis": getattr(diagnosis, "model_dump", lambda: {})() if "diagnosis" in locals() else None,
#                 "response": f"Error correction retry #{retry_count} failed: {str(retry_err)}",
#                 "error": str(retry_err)
#             }
