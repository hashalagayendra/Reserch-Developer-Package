# ErrorCorrectionSubAgent Code (`error_correction_sub_agent.py`)

This file contains the complete source code of `error_correction_sub_agent.py` implementing the recursive self-healing mechanism, error diagnosis, tool discovery integration, replacement agent compilation, and checkpoint updates.

```python
"""
Error Correction Sub-Agent for Main Workflow.

This agent handles recursive error correction and self-healing for sub-agents.
When a sub-agent fails, this agent analyzes the failure, discovers missing tools if needed,
synthesizes a repaired sub-agent configuration, instantiates a replacement sub-agent,
and executes it. If it fails, it recurses up to a maximum of 3 times.
"""

from typing import Optional, List, Any, Dict, Union
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_openrouter import ChatOpenRouter
from research.workflows.tool_descovery_workflow.tool_discovery import (
    ToolDiscoveryWorkflow,
    APIToolDefinition,
)
from research.workflows.main_workflow.utility.transform_tool_discovery_workflow_as_tool import (
    transform_tool_discovery_workflow_as_tool,
)
from research.workflows.main_workflow.utility.create_single_agent import create_single_agent
from research.workflows.main_workflow.schemas import (
    ExecutableSubAgent,
    CheckpointDetails,
    AgentCheckpoint,
    SubAgentExecutionResult,
)
from research.workflows.main_workflow.workflow_logger import WorkflowLogger
from research.workflows.main_workflow.accumulated_context_formatter import (
    format_accumulated_context_from_checkpoints,
)


class RepairedSubAgentConfig(BaseModel):
    """Configuration synthesized by the error correction agent for a replacement sub-agent."""
    diagnosis: str = Field(description="Clear explanation of the error diagnosis and what caused the failure.")
    name: str = Field(description="Name of the replacement sub-agent.")
    system_prompt: str = Field(description="Enhanced, corrected system prompt and instructions fixing previous failure.")
    task: str = Field(description="The task assigned to this replacement sub-agent.")
    model_name: str = Field(default="deepseek/deepseek-v4-flash", description="Model name for the replacement sub-agent.")
    tools: List[APIToolDefinition] = Field(
        default_factory=list,
        description="List of specific API tools (APIToolDefinition) assigned to this replacement sub-agent."
    )


class ErrorCorrectionSubAgent:
    """
    Sub-agent responsible for error correction and recursive self-healing.
    """

    def __init__(
        self,
        tool_discovery_workflow: Optional[ToolDiscoveryWorkflow] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.2
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.tool_discovery_workflow = tool_discovery_workflow

        # Create tool and store as self.tool_discovery_tool (and alias self.tool_discover_tool)
        self.tool_discovery_tool = transform_tool_discovery_workflow_as_tool(self.tool_discovery_workflow)
        self.tool_discover_tool = self.tool_discovery_tool

        # Create agent without passing tools as parameter
        self.error_correction_agent = self.create_error_correction_agent()

    def create_error_correction_agent(self):
        """
        Creates an Error Correction Agent using langchain.agents.create_agent.
        Uses self attributes directly (self.model_name, self.temperature, self.api_key, self.tool_discovery_tool).
        """
        llm = ChatOpenRouter(
            model=self.model_name or "deepseek/deepseek-v4-flash",
            temperature=self.temperature,
            api_key=self.api_key
        )

        system_prompt = (
            "You are an AI Error Correction Agent. Your task is to analyze the failure or error "
            "encountered by a sub-agent and provide a clear, concise description of the error, "
            "identifying what caused the failure and how to resolve it.\n"
            "If missing tools or API limitations caused the error, you have access to the "
            "'tool_discovery_workflow' tool and can invoke it to search and test new API tools."
        )

        tools = [self.tool_discovery_tool] if self.tool_discovery_tool else []

        return create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
            response_format=RepairedSubAgentConfig,
            name="ErrorCorrectionAgent"
        )

    async def start_fix(
        self,
        agent: Union[ExecutableSubAgent, Dict[str, Any]],
        checkpoints: Optional[List[CheckpointDetails]] = None,
        layer_number: Optional[int] = None,
        default_model_name: Optional[str] = None,
        error: Optional[Any] = None,
        user_message: Optional[str] = None,
        response_format: Optional[Any] = None,
        attempt: int = 1,
        max_attempts: int = 3
    ) -> Optional[SubAgentExecutionResult]:
        """
        Recursively fixes a failed sub-agent up to max_attempts (maximum 3 times).
        Generates a replacement sub-agent configuration, instantiates the new sub-agent,
        executes it, and on success logs, updates checkpoints, and returns the result.

        Args:
            agent (Union[ExecutableSubAgent, Dict[str, Any]]): The failed sub-agent instance or config dictionary.
            checkpoints (Optional[List[CheckpointDetails]], optional): List of checkpoint details.
            layer_number (Optional[int], optional): Current execution layer number.
            default_model_name (Optional[str], optional): Fallback model name.
            error (Optional[Any], optional): The exception or error that occurred.
            user_message (Optional[str], optional): User prompt message for execution.
            response_format (Optional[Any], optional): Structured response format if configured.
            attempt (int, optional): Current retry count (starts at 1).
            max_attempts (int, optional): Maximum recursive attempts allowed (default 3).

        Returns:
            Optional[SubAgentExecutionResult]: Successful execution result, or None if attempts exhausted.
        """
        if attempt > max_attempts:
            print(f"[ErrorCorrection] Maximum recursive repair attempts ({max_attempts}) reached.")
            return None

        # 1. Extract agent details into local variables
        agent_data = agent
        checkpoints_data = checkpoints
        layer_number_data = layer_number
        default_model_name_data = default_model_name
        error_data = error

        if isinstance(agent, ExecutableSubAgent):
            agent_name = agent.name
            agent_task = agent.task
            agent_system_prompt = getattr(agent, "system_prompt", "")
            agent_model = agent.model_name
            agent_tools = agent.tools
            agent_executor = agent.executor
            agent_response_format = agent.response_format
            agent_llm = agent.llm
        elif isinstance(agent, dict):
            agent_name = agent.get("name", "SubAgent")
            agent_task = agent.get("task", "")
            agent_system_prompt = agent.get("system_prompt", "")
            agent_model = agent.get("model_name", default_model_name)
            agent_tools = agent.get("tools", [])
            agent_executor = agent.get("executor")
            agent_response_format = agent.get("response_format")
            agent_llm = agent.get("llm")
        else:
            agent_name = getattr(agent, "name", "SubAgent")
            agent_task = getattr(agent, "task", "")
            agent_system_prompt = getattr(agent, "system_prompt", "")
            agent_model = getattr(agent, "model_name", default_model_name)
            agent_tools = getattr(agent, "tools", [])
            agent_executor = getattr(agent, "executor", None)
            agent_response_format = getattr(agent, "response_format", None)
            agent_llm = getattr(agent, "llm", None)

        print("--- [Error Correction Agent] ---")
        print(f"Self-Healing Attempt: {attempt} of {max_attempts}")
        print(f"Error Correction Agent: {self.error_correction_agent}")
        print(f"Agent: {agent_data}")
        print(f"Agent Name: {agent_name}")
        print(f"Agent Task: {agent_task}")
        print(f"Agent Model: {agent_model}")
        print(f"Layer Number: {layer_number_data}")
        print(f"Default Model Name: {default_model_name_data}")
        print(f"Error: {error_data}")
        print(f"Checkpoints: {checkpoints_data}")
        print("--------------------------------")

        # 2. Format accumulated context from previous layers if needed
        accumulated_context = format_accumulated_context_from_checkpoints(
            checkpoints=checkpoints_data,
            layer_number=layer_number_data
        )

        # 3. Diagnose the error and synthesize a replacement sub-agent configuration
        repair_prompt = (
            f"A sub-agent failed during execution in Layer {layer_number_data}.\n\n"
            f"Failed Sub-Agent Details:\n"
            f"- Name: {agent_name}\n"
            f"- Assigned Task: {agent_task}\n"
            f"- Original Instructions: {agent_system_prompt}\n"
            f"- Tools: {agent_tools}\n\n"
            f"Error Encountered:\n{str(error_data)}\n\n"
            f"Accumulated Context from Previous Layers:\n{accumulated_context or 'None'}\n\n"
            f"Self-Healing Attempt: {attempt} of {max_attempts}\n\n"
            "INSTRUCTIONS:\n"
            "1. Diagnose the exact cause of this failure.\n"
            "2. If missing tools caused the failure, you can call 'tool_discovery_workflow' to discover external APIs.\n"
            "3. Generate a corrected, robust sub-agent configuration with enhanced system_prompt and instructions "
            "to successfully accomplish the task and avoid repeating the error."
        )

        repaired_config: Optional[RepairedSubAgentConfig] = None
        try:
            agent_res = await self.error_correction_agent.ainvoke({"messages": [("user", repair_prompt)]})
            structured_res = agent_res.get("structured_response") if isinstance(agent_res, dict) else None
            if isinstance(structured_res, RepairedSubAgentConfig):
                repaired_config = structured_res
            elif isinstance(structured_res, dict):
                repaired_config = RepairedSubAgentConfig(**structured_res)
        except Exception as diag_err:
            print(f"[ErrorCorrection] Diagnosis invoke failed ({diag_err}), using fallback prompt patch.")

        if repaired_config is None:
            repaired_config = RepairedSubAgentConfig(
                diagnosis=f"Fixing failure: {str(error_data)}",
                name=f"{agent_name}_fix",
                system_prompt=(
                    f"{agent_system_prompt}\n\n"
                    f"CRITICAL FIX: The previous execution encountered an error: {str(error_data)}. "
                    f"Adjust your approach and ensure your response succeeds without repeating this error."
                ),
                task=agent_task,
                model_name=agent_model or default_model_name_data or "deepseek/deepseek-v4-flash",
                tools=[]
            )

        # 4. Instantiate the new executable sub-agent using create_single_agent
        resolved_format = response_format or agent_response_format
        replacement_agent: ExecutableSubAgent = create_single_agent(
            model_name=repaired_config.model_name or agent_model or "deepseek/deepseek-v4-flash",
            system_prompt=repaired_config.system_prompt,
            task=repaired_config.task or agent_task,
            name=f"{repaired_config.name}_v{attempt}",
            tools=repaired_config.tools if repaired_config.tools else agent_tools,
            temperature=self.temperature,
            api_key=self.api_key,
            response_format=resolved_format
        )

        # Build prompt message for the replacement agent
        if user_message:
            exec_prompt = user_message
        elif accumulated_context:
            exec_prompt = (
                f"Assigned Task: {replacement_agent.task}\n\n"
                f"Context & Results from Previous Layer(s):\n{accumulated_context}\n\n"
                "Please use the above context and your available tools to accomplish your task."
            )
        else:
            exec_prompt = f"Assigned Task: {replacement_agent.task}"

        WorkflowLogger.log_subagent_start(
            agent_name=replacement_agent.name,
            agent_model=replacement_agent.model_name,
            agent_task=replacement_agent.task,
            layer_number=layer_number_data,
            accumulated_context=accumulated_context
        )

        # 5. Execute the replacement sub-agent
        try:
            res = await replacement_agent.executor.ainvoke({"messages": [("user", exec_prompt)]})

            structured_response = res.get("structured_response") if isinstance(res, dict) else None
            messages = res.get("messages", []) if isinstance(res, dict) else []
            final_content = messages[-1].content if messages else ""

            # If final_content is empty, populate from structured_response
            if not final_content and structured_response is not None:
                if hasattr(structured_response, "model_dump_json"):
                    final_content = structured_response.model_dump_json(indent=2)
                else:
                    final_content = str(structured_response)

            # -----------------------------------------------------------------
            # [LOGGING] Replacement Sub-Agent Execution Completed Successfully
            # -----------------------------------------------------------------
            WorkflowLogger.log_subagent_completed(
                agent_name=replacement_agent.name,
                messages=messages,
                final_content=final_content
            )

            # Record sub-agent checkpoint
            if checkpoints_data is not None:
                tools_used: List[str] = []
                for msg in messages:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tc_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", str(tc))
                            if tc_name and tc_name not in tools_used:
                                tools_used.append(tc_name)

                agent_cp = AgentCheckpoint(
                    agent=replacement_agent.name,
                    status="Success",
                    task=replacement_agent.task,
                    tools=tools_used,
                    output=str(final_content)
                )

                target_layer_cp = next((cp for cp in checkpoints_data if cp.layer_number == layer_number_data), None)
                if target_layer_cp is None:
                    target_layer_cp = CheckpointDetails(layer_number=layer_number_data, agents=[])
                    checkpoints_data.append(target_layer_cp)

                target_layer_cp.agents.append(agent_cp)

            print(f"[ErrorCorrection] Replacement agent succeeded on attempt {attempt}!")

            return SubAgentExecutionResult(
                name=replacement_agent.name,
                task=replacement_agent.task,
                status="success",
                response=str(final_content),
                structured_response=structured_response.model_dump() if hasattr(structured_response, "model_dump") else structured_response,
                messages=messages
            )

        except Exception as new_error:
            print(f"[ErrorCorrection] Replacement agent failed on attempt {attempt}: {new_error}")
            WorkflowLogger.log_subagent_failed(agent_name=replacement_agent.name, error=new_error)

            # 6. RECURSIVE CALL: retry up to max_attempts (maximum 3 times)
            return await self.start_fix(
                agent=replacement_agent,
                checkpoints=checkpoints_data,
                layer_number=layer_number_data,
                default_model_name=default_model_name_data,
                error=new_error,
                user_message=exec_prompt,
                response_format=resolved_format,
                attempt=attempt + 1,
                max_attempts=max_attempts
            )
```
