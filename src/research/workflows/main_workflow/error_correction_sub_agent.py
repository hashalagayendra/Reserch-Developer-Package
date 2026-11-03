"""
Error Correction Sub-Agent for Main Workflow.

This agent handles error correction for sub-agents across 4 specialized levels:
1. Infrastructure
2. Input Prompt & Context
3. Tool Calling
4. Output & Validation

Each node executes a 2-agent architecture:
- Agent 1: Diagnostic / Architecture Generator Agent that generates a new NonExecutableSubAgentConfig.
- Agent 2: Repaired Sub-Agent instantiated via create_single_agent using that specification.

Recovery metrics are logged ONLY during the execution of the new repaired sub-agent (Agent 2),
capturing strictly its success or failure, latency, and token consumption.
"""

import time
import uuid
import json
from typing import Optional, List, Any, Dict, Union, TypedDict

from langchain.agents import create_agent
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

from research.workflows.main_workflow.schemas import (
    NonExecutableSubAgentConfig,
    ExecutableSubAgent,
    CheckpointDetails,
    SubAgentExecutionResult,
)
from research.workflows.main_workflow.utility.transform_tool_discovery_workflow_as_tool import (
    transform_tool_discovery_workflow_as_tool,
)
from research.workflows.main_workflow.utility.create_single_agent import create_single_agent
from research.workflows.main_workflow.accumulated_context_formatter import (
    format_accumulated_context_from_checkpoints,
)
from research.workflows.main_workflow.utility.recovery_experiment_logger import (
    RecoveryExperimentLogger,
)
from research.config import settings
from pydantic import Field


class InfrastructureSubAgentConfig(NonExecutableSubAgentConfig):
    """
    Sub-agent configuration extending NonExecutableSubAgentConfig for Level 1 (Infrastructure) recovery,
    adding the infrastructure provider field as string.
    """
    provider: Optional[str] = Field(
        default=None,
        description="The infrastructure provider to use for the model (e.g. 'Together', 'DeepInfra', 'Fireworks', 'Novita', 'Lepton', 'Hyperbolic', 'Azure', or 'auto')"
    )


class ErrorCorrectionState(TypedDict, total=False):
    """
    State passing through the 4-node Error Correction Graph.
    Maintains experiment_id, agent, and original error code e.
    """
    experiment_id: str
    agent: Any
    e: Any
    error: Any


class ErrorCorrectionSubAgent:
    """
    Sub-agent responsible for error correction and multi-level self-healing.
    """

    def __init__(
        self,
        tool_discovery_workflow: Optional[Any] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.2
    ):
        self.api_key = api_key or settings.OPENROUTER_API_KEY
        self.model_name = model_name or settings.ERROR_CORRECTION_MODEL
        self.temperature = temperature
        self.tool_discovery_workflow = tool_discovery_workflow

        # Convert tool discovery workflow to tool
        self.tool_discovery_tool = transform_tool_discovery_workflow_as_tool(
            self.tool_discovery_workflow
        )

        # Create base diagnostic agent
        self.error_correction_agent = self.create_error_correction_agent()

    def create_error_correction_agent(
        self,
        tools: Optional[List[Any]] = None,
        name: str = "ErrorCorrectionAgent"
    ):
        """
        Creates an Error Correction Agent using langchain.agents.create_agent.
        """
        llm = ChatOpenRouter(
            model=self.model_name or "deepseek/deepseek-v4-flash",
            temperature=self.temperature,
            api_key=self.api_key
        )

        system_prompt = (
            "You are an AI Error Correction Agent. Your task is to analyze the failure or error "
            "encountered by a sub-agent and provide a clear, concise description of the error, "
            "identifying what caused the failure and how to resolve it."
        )

        if tools is not None:
            agent_tools = tools
        else:
            agent_tools = [self.tool_discovery_tool] if getattr(self, "tool_discovery_tool", None) else []

        return create_agent(
            model=llm,
            tools=agent_tools,
            system_prompt=system_prompt,
            name=name
        )

    # -------------------------------------------------------------------------
    # Legacy Fix Function (Maintained exactly as requested)
    # -------------------------------------------------------------------------
    async def start_fix(
        self,
        agent: Union[ExecutableSubAgent, Dict[str, Any]],
        checkpoints: Optional[List[CheckpointDetails]] = None,
        layer_number: Optional[int] = None,
        default_model_name: Optional[str] = None,
        error: Optional[Any] = None
    ) -> Any:
        """
        Receives agent, checkpoints, layer_number, default_model_name, and error.
        Stores all data in variables inside this function and prints them.
        """
        agent_data = agent
        checkpoints_data = checkpoints
        layer_number_data = layer_number
        default_model_name_data = default_model_name
        error_data = error

        if isinstance(agent, ExecutableSubAgent):
            agent_name = agent.name
            agent_task = agent.task
            agent_model = agent.model_name
        elif isinstance(agent, dict):
            agent_name = agent.get("name", "SubAgent")
            agent_task = agent.get("task", "")
            agent_model = agent.get("model_name", default_model_name)
        else:
            agent_name = getattr(agent, "name", "SubAgent")
            agent_task = getattr(agent, "task", "")
            agent_model = getattr(agent, "model_name", default_model_name)

        print("--- [Error Correction Agent] ---")
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
        return error_data

    # -------------------------------------------------------------------------
    # Helper: Extract Token Usage from Sub-Agent execution
    # -------------------------------------------------------------------------
    def _extract_tokens(self, res: Any, user_message: str) -> (int, int):
        """Extracts or approximates token counts from LangChain execution result."""
        prompt_tokens = 0
        completion_tokens = 0
        if isinstance(res, dict):
            messages = res.get("messages", [])
            for msg in messages:
                if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                    prompt_tokens += msg.usage_metadata.get("input_tokens", 0)
                    completion_tokens += msg.usage_metadata.get("output_tokens", 0)
                elif hasattr(msg, "response_metadata") and isinstance(msg.response_metadata, dict):
                    usage = msg.response_metadata.get("token_usage", {})
                    prompt_tokens += usage.get("prompt_tokens", 0)
                    completion_tokens += usage.get("completion_tokens", 0)

        if prompt_tokens == 0:
            prompt_tokens = max(len(user_message.split()) * 2, 50)
        if completion_tokens == 0:
            completion_tokens = 120

        return prompt_tokens, completion_tokens

    # -------------------------------------------------------------------------
    # Main Function: get_error_details (Executes 4-node Graph with Minimal State)
    # -------------------------------------------------------------------------
    async def get_error_details(
        self,
        agent: Union[ExecutableSubAgent, Dict[str, Any]],
        checkpoints: Optional[List[CheckpointDetails]] = None,
        layer_number: Optional[int] = None,
        default_model_name: Optional[str] = None,
        error: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Executes the 4-node Error Correction Graph:
        1. Infrastructure
        2. Input Prompt & Context
        3. Tool Calling
        4. Output & Validation

        Each node generates configuration via Agent 1, and logs recovery metrics
        ONLY when running the newly instantiated sub-agent (Agent 2).
        """
        experiment_id = f"EXP-{int(time.time())}-{uuid.uuid4().hex[:6].upper()}"

        # ---------------------------------------------------------------------
        # Define Node 1: Infrastructure Node
        # ---------------------------------------------------------------------
        async def infrastructure_node(state: ErrorCorrectionState) -> ErrorCorrectionState:
            agent_start_time = time.time()
            agent = state.get("agent")
            if isinstance(agent, dict):
                agent_name = agent.get("name", "SubAgent")
                agent_task = agent.get("task", "")
                agent_tools = agent.get("tools") or []
                agent_response_format = agent.get("response_format")
            elif agent is not None:
                agent_name = agent.name
                agent_task = agent.task
                agent_tools = agent.tools or []
                agent_response_format = agent.response_format
            else:
                agent_name = "SubAgent"
                agent_task = ""
                agent_tools = []
                agent_response_format = None

            accumulated_context = format_accumulated_context_from_checkpoints(
                checkpoints=checkpoints,
                layer_number=layer_number
            )
            if accumulated_context:
                user_message = (
                    f"Assigned Task: {agent_task}\n\n"
                    f"Context & Results from Previous Layer(s):\n{accumulated_context}\n\n"
                    "Please use the above context and your available tools to accomplish your task."
                )
            else:
                user_message = f"Assigned Task: {agent_task}"

            generator_model = "deepseek/deepseek-v4-flash"
            node_name = "InfrastructureSpecialist"
            repair_level = "Level 1 (Infrastructure)"
            recovery_method = "Model Fallback & System Parameter Adaptation"
            generator_temperature = 0.1
            fallback_providers = [
                "Together",
                "DeepInfra",
                "Fireworks",
                "Novita",
                "Lepton",
                "Hyperbolic",
                "Azure"
            ]
            generator_system_prompt = (
                "You are an AI Architecture & Infrastructure Recovery Specialist Agent operating within a 4-level recovery hierarchy:\n"
                "- Level 1: Infrastructure (Model fallback, endpoint and infrastructure provider selection, rate limits, execution parameters)\n"
                "- Level 2: Input Prompt & Context (Prompt instruction refinement, context trimming, ambiguity resolution)\n"
                "- Level 3: Tool Calling (Dynamic tool discovery, argument correction, tool schema mapping)\n"
                "- Level 4: Output & Validation (Output format enforcement, JSON schema alignment, Pydantic validation)\n\n"
                "CRITICAL INSTRUCTION FOR THIS NODE:\n"
                "You are operating strictly at Level 1 (Infrastructure). When generating the corrected sub-agent specification "
                "(InfrastructureSubAgentConfig), you must ONLY change infrastructure-related configuration (such as model_name or the infrastructure provider). "
                "Do NOT change or attempt to fix the other levels during this step: do NOT rewrite prompt instructions (Level 2), "
                "do NOT modify tool configurations (Level 3), and do NOT modify output schemas (Level 4). "
                "Preserve the original task and tools as-is, focusing exclusively on infrastructure recovery.\n\n"
                "INFRASTRUCTURE PROVIDERS FOR ROUTING:\n"
                f"When selecting or changing the backend infrastructure provider, set the 'provider' field to a valid provider string if needed "
                f"(supported providers: {', '.join(fallback_providers)}, or None/'auto' for default routing).\n\n"
                "AVAILABLE MODELS FOR FALLBACK / MODEL REPLACEMENT:\n"
                "When changing the LLM model is required (e.g., due to model rate limits, context overflow, downtime, or capability issues), "
                "select the most suitable model from this list for model_name:\n"
                "- 'deepseek/deepseek-v4-flash' (Default, ultra-fast, high cost-efficiency)\n"
                "- 'deepseek/deepseek-chat' (General reasoning and strong instruction-following)\n"
                "- 'openai/gpt-4o-mini' (Fast, robust instruction adherence, high throughput)\n"
                "- 'openai/gpt-4o' (Advanced multi-step reasoning and complex tasks)\n"
                "- 'anthropic/claude-3.5-sonnet' (Top-tier reasoning, code generation, and complex workflows)\n"
                "- 'anthropic/claude-3-haiku' (Ultra-low latency, lightweight fallback)\n"
                "- 'google/gemini-2.0-flash-001' (High context window, multimodal capabilities, fast response)\n"
                "- 'meta-llama/llama-3.3-70b-instruct' (Open-weights, reliable general performance)\n"
                "- 'qwen/qwen-2.5-72b-instruct' (Strong analytical and multilingual capabilities)\n"
                "- 'mistralai/mistral-large-2411' (High context adherence, complex logic)\n\n"
                "Only change model_name from this list if changing the model is necessary to resolve the infrastructure error.\n\n"
                "RESEARCH CONTEXT & EXPERIMENTAL NATURE:\n"
                "This error-correction pipeline is an academic research benchmark designed to collect empirical recovery data. "
                "It is completely acceptable and expected that some proposed repairs or configurations may fail—that is not a problem. "
                "Do not hallucinate impossible configurations or over-engineer in an artificial attempt to guarantee success. "
                "Formulate the most plausible infrastructure hypothesis; both successes and failures provide vital research insights."
            )

            err = state.get("e", state.get("error"))
            error_code = str(err) if err is not None else "UnknownError"
            exp_id = state.get("experiment_id", "")
            current_model = getattr(agent, "model_name", "deepseek/deepseek-v4-flash") if not isinstance(agent, dict) else agent.get("model_name", "deepseek/deepseek-v4-flash")

            # 1. Agent 1: Generate Configuration (no logging for spec_agent)
            try:
                generator_llm = ChatOpenRouter(
                    model=generator_model,
                    temperature=generator_temperature,
                    api_key=self.api_key
                )
                spec_agent = create_react_agent(
                    model=generator_llm,
                    tools=[],
                    prompt=generator_system_prompt,
                    response_format=InfrastructureSubAgentConfig,
                    name="InfraSpecGeneratorAgent"
                )
                generator_user_msg = (
                    f"Sub-agent execution failed.\n"
                    f"- Agent Name: {agent_name}\n"
                    f"- Current Model: {current_model}\n"
                    f"- Assigned Task: {agent_task}\n"
                    f"- Failure / Error Code: {error_code}\n"
                    f"- Error Details: {str(err)}\n\n"
                    "Generate the corrected InfrastructureSubAgentConfig to execute this task successfully. "
                    "Select a replacement model and/or infrastructure provider from the available list if needed."
                )
                spec_res = await spec_agent.ainvoke({"messages": [("user", generator_user_msg)]})
                structured_spec = spec_res.get("structured_response")
                if isinstance(structured_spec, InfrastructureSubAgentConfig):
                    spec = structured_spec
                elif isinstance(structured_spec, dict):
                    spec = InfrastructureSubAgentConfig(**structured_spec)
                elif isinstance(structured_spec, NonExecutableSubAgentConfig):
                    spec = InfrastructureSubAgentConfig(**structured_spec.model_dump())
                else:
                    print(f"[{node_name}] Spec generation did not return a valid configuration.")
                    return state
            except Exception as gen_err:
                print(f"[{node_name}] Failed generating agent execution: {gen_err}")
                return state

            used_model = spec.model_name or generator_model
            used_provider = getattr(spec, "provider", None)
            if used_provider:
                recovery_method = f"Model & Provider Adaptation ({used_model} via {used_provider})"
            else:
                recovery_method = f"Model Fallback & Parameter Adaptation ({used_model})"

            # 2. Agent 2: Create Sub-Agent via create_single_agent (No tools for infrastructure level)
            sub_agent_instance = create_single_agent(
                model_name=used_model,
                system_prompt=spec.system_prompt,
                task=spec.task or agent_task,
                name=spec.name or f"{agent_name}_{node_name}",
                tools=[],
                temperature=generator_temperature,
                api_key=self.api_key,
                response_format=agent_response_format,
                provider=used_provider
            )

            # 3. Execute New Agent & Log ONLY for this new agent execution
            try:
                res = await sub_agent_instance.executor.ainvoke({"messages": [("user", user_message)]})
                agent_elapsed = time.time() - agent_start_time
                agent_p_toks, agent_c_toks = self._extract_tokens(res, user_message)

                RecoveryExperimentLogger.log_recovery_attempt(
                    experiment_id=exp_id,
                    error_code=error_code,
                    error_category="Infrastructure",
                    repair_level=repair_level,
                    recovery_method=recovery_method,
                    success=True,
                    recovery_time_seconds=agent_elapsed,
                    prompt_tokens=agent_p_toks,
                    completion_tokens=agent_c_toks,
                    model_name=used_model
                )
            except Exception as agent_err:
                agent_elapsed = time.time() - agent_start_time
                RecoveryExperimentLogger.log_recovery_attempt(
                    experiment_id=exp_id,
                    error_code=error_code,
                    error_category="Infrastructure",
                    repair_level=repair_level,
                    recovery_method=recovery_method,
                    success=False,
                    recovery_time_seconds=agent_elapsed,
                    prompt_tokens=50,
                    completion_tokens=10,
                    model_name=used_model
                )

            return state

        # ---------------------------------------------------------------------
        # Define Node 2: Input Prompt & Context Node
        # ---------------------------------------------------------------------
        async def prompt_context_node(state: ErrorCorrectionState) -> ErrorCorrectionState:
            agent_start_time = time.time()
            agent = state.get("agent")
            if isinstance(agent, dict):
                agent_name = agent.get("name", "SubAgent")
                agent_task = agent.get("task", "")
                agent_tools = agent.get("tools") or []
                agent_response_format = agent.get("response_format")
            elif agent is not None:
                agent_name = agent.name
                agent_task = agent.task
                agent_tools = agent.tools or []
                agent_response_format = agent.response_format
            else:
                agent_name = "SubAgent"
                agent_task = ""
                agent_tools = []
                agent_response_format = None

            accumulated_context = format_accumulated_context_from_checkpoints(
                checkpoints=checkpoints,
                layer_number=layer_number
            )
            if accumulated_context:
                user_message = (
                    f"Assigned Task: {agent_task}\n\n"
                    f"Context & Results from Previous Layer(s):\n{accumulated_context}\n\n"
                    "Please use the above context and your available tools to accomplish your task."
                )
            else:
                user_message = f"Assigned Task: {agent_task}"

            generator_model = "deepseek/deepseek-v4-flash"
            node_name = "PromptContextSpecialist"
            repair_level = "Level 2 (Input Prompt & Context)"
            recovery_method = "Prompt Instruction Refinement & Context Sanitization"
            generator_temperature = 0.2
            generator_system_prompt = (
                "You are an AI Prompt Engineering & Context Recovery Specialist Agent operating within a 4-level recovery hierarchy:\n"
                "- Level 1: Infrastructure (Model fallback, endpoint selection, rate limits, execution parameters)\n"
                "- Level 2: Input Prompt & Context (Prompt instruction refinement, context trimming, ambiguity resolution)\n"
                "- Level 3: Tool Calling (Dynamic tool discovery, argument correction, tool schema mapping)\n"
                "- Level 4: Output & Validation (Output format enforcement, JSON schema alignment, Pydantic validation)\n\n"
                "CRITICAL INSTRUCTION FOR THIS NODE:\n"
                "You are operating strictly at Level 2 (Input Prompt & Context). When generating the corrected sub-agent specification "
                "(NonExecutableSubAgentConfig), you must ONLY change prompt and context-related configurations (such as rewriting the system_prompt, "
                "clarifying task instructions, disambiguating constraints, and pruning noisy context). "
                "Do NOT change or attempt to fix the other levels during this step: do NOT change model_name (Level 1), "
                "do NOT modify tool configurations or tool discovery (Level 3), and do NOT alter output validation schemas (Level 4). "
                "Preserve the existing model_name and tools as-is, focusing exclusively on input prompt and context engineering to eliminate the failure.\n\n"
                "RESEARCH CONTEXT & EXPERIMENTAL NATURE:\n"
                "This error-correction pipeline is an academic research benchmark designed to collect empirical recovery data. "
                "It is completely acceptable and expected that some proposed repairs or configurations may fail—that is not a problem. "
                "Do not hallucinate impossible constraints or over-engineer in an artificial attempt to guarantee success. "
                "Formulate the most plausible prompt-engineering hypothesis; both successes and failures provide vital research insights."
            )

            err = state.get("e", state.get("error"))
            error_code = str(err) if err is not None else "UnknownError"
            exp_id = state.get("experiment_id", "")

            # 1. Agent 1: Generate Configuration (no logging for spec_agent)
            try:
                generator_llm = ChatOpenRouter(
                    model=generator_model,
                    temperature=generator_temperature,
                    api_key=self.api_key
                )
                spec_agent = create_react_agent(
                    model=generator_llm,
                    tools=[],
                    prompt=generator_system_prompt,
                    response_format=NonExecutableSubAgentConfig,
                    name="PromptSpecGeneratorAgent"
                )
                generator_user_msg = (
                    f"Sub-agent execution failed.\n"
                    f"- Agent Name: {agent_name}\n"
                    f"- Assigned Task: {agent_task}\n"
                    f"- Failure / Error Code: {error_code}\n"
                    f"- Error Details: {str(err)}\n\n"
                    "Generate the corrected NonExecutableSubAgentConfig rewriting instructions to avoid this error."
                )
                spec_res = await spec_agent.ainvoke({"messages": [("user", generator_user_msg)]})
                structured_spec = spec_res.get("structured_response")
                if isinstance(structured_spec, NonExecutableSubAgentConfig):
                    spec = structured_spec
                elif isinstance(structured_spec, dict):
                    spec = NonExecutableSubAgentConfig(**structured_spec)
                else:
                    print(f"[{node_name}] Spec generation did not return a valid configuration.")
                    return state
            except Exception as gen_err:
                print(f"[{node_name}] Failed generating agent execution: {gen_err}")
                return state

            used_model = spec.model_name or generator_model

            # 2. Agent 2: Create Sub-Agent via create_single_agent (No tools for prompt & context level)
            sub_agent_instance = create_single_agent(
                model_name=used_model,
                system_prompt=spec.system_prompt,
                task=spec.task or agent_task,
                name=spec.name or f"{agent_name}_{node_name}",
                tools=[],
                temperature=generator_temperature,
                api_key=self.api_key,
                response_format=agent_response_format
            )

            # 3. Execute New Agent & Log ONLY for this new agent execution
            try:
                res = await sub_agent_instance.executor.ainvoke({"messages": [("user", user_message)]})
                agent_elapsed = time.time() - agent_start_time
                agent_p_toks, agent_c_toks = self._extract_tokens(res, user_message)

                RecoveryExperimentLogger.log_recovery_attempt(
                    experiment_id=exp_id,
                    error_code=error_code,
                    error_category="Input Prompt & Context",
                    repair_level=repair_level,
                    recovery_method=recovery_method,
                    success=True,
                    recovery_time_seconds=agent_elapsed,
                    prompt_tokens=agent_p_toks,
                    completion_tokens=agent_c_toks,
                    model_name=used_model
                )
            except Exception as agent_err:
                agent_elapsed = time.time() - agent_start_time
                RecoveryExperimentLogger.log_recovery_attempt(
                    experiment_id=exp_id,
                    error_code=error_code,
                    error_category="Input Prompt & Context",
                    repair_level=repair_level,
                    recovery_method=recovery_method,
                    success=False,
                    recovery_time_seconds=agent_elapsed,
                    prompt_tokens=50,
                    completion_tokens=10,
                    model_name=used_model
                )

            return state

        # ---------------------------------------------------------------------
        # Define Node 3: Tool Calling Node
        # ---------------------------------------------------------------------
        async def tool_calling_node(state: ErrorCorrectionState) -> ErrorCorrectionState:
            agent_start_time = time.time()
            agent = state.get("agent")
            if isinstance(agent, dict):
                agent_name = agent.get("name", "SubAgent")
                agent_task = agent.get("task", "")
                agent_tools = agent.get("tools") or []
                agent_response_format = agent.get("response_format")
            elif agent is not None:
                agent_name = agent.name
                agent_task = agent.task
                agent_tools = agent.tools or []
                agent_response_format = agent.response_format
            else:
                agent_name = "SubAgent"
                agent_task = ""
                agent_tools = []
                agent_response_format = None

            accumulated_context = format_accumulated_context_from_checkpoints(
                checkpoints=checkpoints,
                layer_number=layer_number
            )
            if accumulated_context:
                user_message = (
                    f"Assigned Task: {agent_task}\n\n"
                    f"Context & Results from Previous Layer(s):\n{accumulated_context}\n\n"
                    "Please use the above context and your available tools to accomplish your task."
                )
            else:
                user_message = f"Assigned Task: {agent_task}"

            generator_model = "deepseek/deepseek-v4-flash"
            node_name = "ToolCallingSpecialist"
            repair_level = "Level 3 (Tool Calling)"
            recovery_method = "Dynamic Tool Discovery & Parameter Mapping"
            generator_temperature = 0.1
            generator_system_prompt = (
                "You are an AI Tool-Calling & Schema Recovery Specialist Agent operating within a 4-level recovery hierarchy:\n"
                "- Level 1: Infrastructure (Model fallback, endpoint selection, rate limits, execution parameters)\n"
                "- Level 2: Input Prompt & Context (Prompt instruction refinement, context trimming, ambiguity resolution)\n"
                "- Level 3: Tool Calling (Dynamic tool discovery, argument correction, tool schema mapping)\n"
                "- Level 4: Output & Validation (Output format enforcement, JSON schema alignment, Pydantic validation)\n\n"
                "CRITICAL INSTRUCTION FOR THIS NODE:\n"
                "You are operating strictly at Level 3 (Tool Calling). When generating the corrected sub-agent specification "
                "(NonExecutableSubAgentConfig), you must ONLY change tool-calling aspects (such as configuring appropriate tools in the tools list, "
                "enabling dynamic tool discovery, or adding explicit tool usage instructions and parameter constraints). "
                "Do NOT change or attempt to fix the other levels during this step: do NOT change model_name (Level 1), "
                "do NOT rewrite the overall task formulation (Level 2), and do NOT modify output validation schemas (Level 4). "
                "Focus exclusively on tool selection, parameter alignment, and accurate tool invocation schemas to resolve the error.\n\n"
                "RESEARCH CONTEXT & EXPERIMENTAL NATURE:\n"
                "This error-correction pipeline is an academic research benchmark designed to collect empirical recovery data. "
                "It is completely acceptable and expected that some proposed repairs or configurations may fail—that is not a problem. "
                "Do not hallucinate non-existent tools or over-engineer in an artificial attempt to guarantee success. "
                "Formulate the most plausible tool-calling hypothesis; both successes and failures provide vital research insights."
            )

            err = state.get("e", state.get("error"))
            error_code = str(err) if err is not None else "UnknownError"
            exp_id = state.get("experiment_id", "")

            # Only tool recovery node uses the tool discovery/generation workflow as tool
            discovery_tools = [self.tool_discovery_tool] if getattr(self, "tool_discovery_tool", None) else []

            # 1. Agent 1: Generate Configuration (no logging for spec_agent)
            try:
                generator_llm = ChatOpenRouter(
                    model=generator_model,
                    temperature=generator_temperature,
                    api_key=self.api_key
                )
                spec_agent = create_react_agent(
                    model=generator_llm,
                    tools=discovery_tools,
                    prompt=generator_system_prompt,
                    response_format=NonExecutableSubAgentConfig,
                    name="ToolSpecGeneratorAgent"
                )
                generator_user_msg = (
                    f"Sub-agent execution failed during tool calling.\n"
                    f"- Agent Name: {agent_name}\n"
                    f"- Assigned Task: {agent_task}\n"
                    f"- Failure / Error Code: {error_code}\n"
                    f"- Error Details: {str(err)}\n\n"
                    "Generate the corrected NonExecutableSubAgentConfig ensuring accurate tool invocations."
                )
                spec_res = await spec_agent.ainvoke({"messages": [("user", generator_user_msg)]})
                structured_spec = spec_res.get("structured_response")
                if isinstance(structured_spec, NonExecutableSubAgentConfig):
                    spec = structured_spec
                elif isinstance(structured_spec, dict):
                    spec = NonExecutableSubAgentConfig(**structured_spec)
                else:
                    print(f"[{node_name}] Spec generation did not return a valid configuration.")
                    return state
            except Exception as gen_err:
                print(f"[{node_name}] Failed generating agent execution: {gen_err}")
                return state

            used_model = spec.model_name or generator_model

            # 2. Agent 2: Create Sub-Agent with tool discovery workflow as tool
            repaired_tools = list(discovery_tools)
            if spec.tools:
                for t in spec.tools:
                    if t not in repaired_tools:
                        repaired_tools.append(t)

            sub_agent_instance = create_single_agent(
                model_name=used_model,
                system_prompt=spec.system_prompt,
                task=spec.task or agent_task,
                name=spec.name or f"{agent_name}_{node_name}",
                tools=repaired_tools,
                temperature=generator_temperature,
                api_key=self.api_key,
                response_format=agent_response_format
            )

            # 3. Execute New Agent & Log ONLY for this new agent execution
            try:
                res = await sub_agent_instance.executor.ainvoke({"messages": [("user", user_message)]})
                agent_elapsed = time.time() - agent_start_time
                agent_p_toks, agent_c_toks = self._extract_tokens(res, user_message)

                RecoveryExperimentLogger.log_recovery_attempt(
                    experiment_id=exp_id,
                    error_code=error_code,
                    error_category="Tool Calling",
                    repair_level=repair_level,
                    recovery_method=recovery_method,
                    success=True,
                    recovery_time_seconds=agent_elapsed,
                    prompt_tokens=agent_p_toks,
                    completion_tokens=agent_c_toks,
                    model_name=used_model
                )
            except Exception as agent_err:
                agent_elapsed = time.time() - agent_start_time
                RecoveryExperimentLogger.log_recovery_attempt(
                    experiment_id=exp_id,
                    error_code=error_code,
                    error_category="Tool Calling",
                    repair_level=repair_level,
                    recovery_method=recovery_method,
                    success=False,
                    recovery_time_seconds=agent_elapsed,
                    prompt_tokens=50,
                    completion_tokens=10,
                    model_name=used_model
                )

            return state

        # ---------------------------------------------------------------------
        # Define Node 4: Output & Validation Node
        # ---------------------------------------------------------------------
        async def output_validation_node(state: ErrorCorrectionState) -> ErrorCorrectionState:
            agent_start_time = time.time()
            agent = state.get("agent")
            if isinstance(agent, dict):
                agent_name = agent.get("name", "SubAgent")
                agent_task = agent.get("task", "")
                agent_tools = agent.get("tools") or []
                agent_response_format = agent.get("response_format")
            elif agent is not None:
                agent_name = agent.name
                agent_task = agent.task
                agent_tools = agent.tools or []
                agent_response_format = agent.response_format
            else:
                agent_name = "SubAgent"
                agent_task = ""
                agent_tools = []
                agent_response_format = None

            accumulated_context = format_accumulated_context_from_checkpoints(
                checkpoints=checkpoints,
                layer_number=layer_number
            )
            if accumulated_context:
                user_message = (
                    f"Assigned Task: {agent_task}\n\n"
                    f"Context & Results from Previous Layer(s):\n{accumulated_context}\n\n"
                    "Please use the above context and your available tools to accomplish your task."
                )
            else:
                user_message = f"Assigned Task: {agent_task}"

            generator_model = "deepseek/deepseek-v4-flash"
            node_name = "OutputValidationSpecialist"
            repair_level = "Level 4 (Output & Validation)"
            recovery_method = "Output Schema Alignment & Structured Repair"
            generator_temperature = 0.0
            generator_system_prompt = (
                "You are an AI Output Schema & Validation Recovery Specialist Agent operating within a 4-level recovery hierarchy:\n"
                "- Level 1: Infrastructure (Model fallback, endpoint selection, rate limits, execution parameters)\n"
                "- Level 2: Input Prompt & Context (Prompt instruction refinement, context trimming, ambiguity resolution)\n"
                "- Level 3: Tool Calling (Dynamic tool discovery, argument correction, tool schema mapping)\n"
                "- Level 4: Output & Validation (Output format enforcement, JSON schema alignment, Pydantic validation)\n\n"
                "CRITICAL INSTRUCTION FOR THIS NODE:\n"
                "You are operating strictly at Level 4 (Output & Validation). When generating the corrected sub-agent specification "
                "(NonExecutableSubAgentConfig), you must ONLY change output formatting and validation adherence instructions "
                "(such as instructing the agent on exact JSON keys, data types, schema validation rules, and eliminating extra markdown or conversational text). "
                "Do NOT change or attempt to fix the other levels during this step: do NOT change model_name (Level 1), "
                "do NOT change task scope (Level 2), and do NOT alter tool definitions (Level 3). "
                "Focus exclusively on enforcing strict output schema compliance, parsing validity, and structured response conformance.\n\n"
                "RESEARCH CONTEXT & EXPERIMENTAL NATURE:\n"
                "This error-correction pipeline is an academic research benchmark designed to collect empirical recovery data. "
                "It is completely acceptable and expected that some proposed repairs or configurations may fail—that is not a problem. "
                "Do not hallucinate output schemas or over-engineer in an artificial attempt to guarantee success. "
                "Formulate the most plausible output validation hypothesis; both successes and failures provide vital research insights."
            )

            err = state.get("e", state.get("error"))
            error_code = str(err) if err is not None else "UnknownError"
            exp_id = state.get("experiment_id", "")

            # 1. Agent 1: Generate Configuration (no logging for spec_agent)
            try:
                generator_llm = ChatOpenRouter(
                    model=generator_model,
                    temperature=generator_temperature,
                    api_key=self.api_key
                )
                spec_agent = create_react_agent(
                    model=generator_llm,
                    tools=[],
                    prompt=generator_system_prompt,
                    response_format=NonExecutableSubAgentConfig,
                    name="OutputSpecGeneratorAgent"
                )
                generator_user_msg = (
                    f"Sub-agent execution failed during output validation.\n"
                    f"- Agent Name: {agent_name}\n"
                    f"- Assigned Task: {agent_task}\n"
                    f"- Failure / Error Code: {error_code}\n"
                    f"- Error Details: {str(err)}\n\n"
                    "Generate the corrected NonExecutableSubAgentConfig enforcing strict schema compliance."
                )
                spec_res = await spec_agent.ainvoke({"messages": [("user", generator_user_msg)]})
                structured_spec = spec_res.get("structured_response")
                if isinstance(structured_spec, NonExecutableSubAgentConfig):
                    spec = structured_spec
                elif isinstance(structured_spec, dict):
                    spec = NonExecutableSubAgentConfig(**structured_spec)
                else:
                    print(f"[{node_name}] Spec generation did not return a valid configuration.")
                    return state
            except Exception as gen_err:
                print(f"[{node_name}] Failed generating agent execution: {gen_err}")
                return state

            used_model = spec.model_name or generator_model

            # 2. Agent 2: Create Sub-Agent via create_single_agent (No tools for output validation level)
            sub_agent_instance = create_single_agent(
                model_name=used_model,
                system_prompt=spec.system_prompt,
                task=spec.task or agent_task,
                name=spec.name or f"{agent_name}_{node_name}",
                tools=[],
                temperature=generator_temperature,
                api_key=self.api_key,
                response_format=agent_response_format
            )

            # 3. Execute New Agent & Log ONLY for this new agent execution
            try:
                res = await sub_agent_instance.executor.ainvoke({"messages": [("user", user_message)]})
                agent_elapsed = time.time() - agent_start_time
                agent_p_toks, agent_c_toks = self._extract_tokens(res, user_message)

                RecoveryExperimentLogger.log_recovery_attempt(
                    experiment_id=exp_id,
                    error_code=error_code,
                    error_category="Output & Validation",
                    repair_level=repair_level,
                    recovery_method=recovery_method,
                    success=True,
                    recovery_time_seconds=agent_elapsed,
                    prompt_tokens=agent_p_toks,
                    completion_tokens=agent_c_toks,
                    model_name=used_model
                )
            except Exception as agent_err:
                agent_elapsed = time.time() - agent_start_time
                RecoveryExperimentLogger.log_recovery_attempt(
                    experiment_id=exp_id,
                    error_code=error_code,
                    error_category="Output & Validation",
                    repair_level=repair_level,
                    recovery_method=recovery_method,
                    success=False,
                    recovery_time_seconds=agent_elapsed,
                    prompt_tokens=50,
                    completion_tokens=10,
                    model_name=used_model
                )

            return state

        # ---------------------------------------------------------------------
        # Build & Execute 4-Node Recovery Graph for this execution
        # ---------------------------------------------------------------------
        workflow = StateGraph(ErrorCorrectionState)
        workflow.add_node("infrastructure_node", infrastructure_node)
        workflow.add_node("prompt_context_node", prompt_context_node)
        workflow.add_node("tool_calling_node", tool_calling_node)
        workflow.add_node("output_validation_node", output_validation_node)

        workflow.add_edge(START, "infrastructure_node")
        workflow.add_edge("infrastructure_node", "prompt_context_node")
        workflow.add_edge("prompt_context_node", "tool_calling_node")
        workflow.add_edge("tool_calling_node", "output_validation_node")
        workflow.add_edge("output_validation_node", END)

        recovery_graph = workflow.compile()

        # Initialize state with experiment_id, agent, and error
        initial_state: ErrorCorrectionState = {
            "experiment_id": experiment_id,
            "agent": agent,
            "e": error,
            "error": error
        }

        final_state = await recovery_graph.ainvoke(initial_state)

        return final_state
