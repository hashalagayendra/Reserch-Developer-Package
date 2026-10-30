import asyncio
from typing import List, Dict, Any, Union, Optional
from research.workflows.main_workflow.schemas import (
    ExecutableSubAgent,
    ExecutableLayer,
    ExecutableLayers,
    SubAgentExecutionResult,
    LayerExecutionResult,
    ExecutionSummary,
    AgentCheckpoint,
    CheckpointDetails,
)
from research.workflows.main_workflow.workflow_logger import WorkflowLogger
from research.workflows.main_workflow.accumulated_context_formatter import format_accumulated_context_from_checkpoints


async def run_single_agent(
    agent: Union[ExecutableSubAgent, Dict[str, Any]],
    checkpoints: Optional[List[CheckpointDetails]] = None,
    layer_number: Optional[int] = None,
    default_model_name: Optional[str] = None,
    error_correction_agent: Optional[Any] = None
) -> SubAgentExecutionResult:
    """
    Executes a single sub-agent using its compiled LangGraph executor.
    
    Args:
        agent (Union[ExecutableSubAgent, Dict[str, Any]]): The executable sub-agent instance or config dictionary.
        checkpoints (Optional[List[CheckpointDetails]], optional): The list of checkpoint details from previous layers.
        layer_number (Optional[int], optional): Current execution layer number.
        default_model_name (Optional[str], optional): Fallback model name if not present in agent.
        error_correction_agent (Optional[Any], optional): Sub-agent for error diagnosis and recovery.
        
    Returns:
        SubAgentExecutionResult: The result of the sub-agent execution containing status, response, messages, and error if any.
    """
    fallback_model = default_model_name
    if isinstance(agent, ExecutableSubAgent):
        agent_name = agent.name
        agent_task = agent.task
        agent_model = agent.model_name
        executor = agent.executor
        response_format = agent.response_format
        agent_llm = agent.llm
    elif isinstance(agent, dict):
        agent_name = agent.get("name", "SubAgent")
        agent_task = agent.get("task", "")
        agent_model = agent.get("model_name", fallback_model)
        executor = agent.get("executor")
        response_format = agent.get("response_format")
        agent_llm = agent.get("llm")
    else:
        agent_name = getattr(agent, "name", "SubAgent")
        agent_task = getattr(agent, "task", "")
        agent_model = getattr(agent, "model_name", fallback_model)
        executor = getattr(agent, "executor", None)
        response_format = getattr(agent, "response_format", None)
        agent_llm = getattr(agent, "llm", None)

    # Build user prompt combining agent's task with accumulated context from previous layers
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

    # -------------------------------------------------------------------------
    # [LOGGING] Sub-Agent Execution Starting
    # -------------------------------------------------------------------------
    WorkflowLogger.log_subagent_start(
        agent_name=agent_name,
        agent_model=agent_model,
        agent_task=agent_task,
        layer_number=layer_number,
        accumulated_context=accumulated_context
    )

    try:
        res = await executor.ainvoke({"messages": [("user", user_message)]})
        
        # Extract structured response directly from the agent execution result
        structured_response = res.get("structured_response") if isinstance(res, dict) else None
        messages = res.get("messages", []) if isinstance(res, dict) else []
        final_content = messages[-1].content if messages else ""

        # Check if any tool message returned an error during the agent execution invoke
        for msg in messages:
            if getattr(msg, "status", None) == "error":
                if error_correction_agent is not None:
                    try:
                        if hasattr(error_correction_agent, "start_fix"):
                            fix_res = error_correction_agent.start_fix(str(getattr(msg, "content", "")))
                            if asyncio.iscoroutine(fix_res):
                                await fix_res
                    except Exception as ec_err:
                        WorkflowLogger.log_warning(f"Error correction agent failed on tool error message: {ec_err}")

        # If final_content is empty (often happens when structured_response is used), populate from structured_response
        if not final_content and structured_response is not None:
            if hasattr(structured_response, "model_dump_json"):
                final_content = structured_response.model_dump_json(indent=2)
            else:
                final_content = str(structured_response)

        # ---------------------------------------------------------------------
        # [LOGGING] Sub-Agent Execution Completed Successfully
        # ---------------------------------------------------------------------
        WorkflowLogger.log_subagent_completed(
            agent_name=agent_name,
            messages=messages,
            final_content=final_content
        )

        # Record sub-agent checkpoint
        if checkpoints is not None:
            tools_used: List[str] = []
            for msg in messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tc_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", str(tc))
                        if tc_name and tc_name not in tools_used:
                            tools_used.append(tc_name)

            agent_cp = AgentCheckpoint(
                agent=agent_name,
                status="Success",
                task=agent_task,
                tools=tools_used,
                output=str(final_content)
            )
            target_layer_cp = next((cp for cp in checkpoints if cp.layer_number == layer_number), None)
            if target_layer_cp is None:
                target_layer_cp = CheckpointDetails(layer_number=layer_number, agents=[])
                checkpoints.append(target_layer_cp)
            target_layer_cp.agents.append(agent_cp)

        return SubAgentExecutionResult(
            name=agent_name,
            task=agent_task,
            status="success",
            response=str(final_content),
            structured_response=structured_response.model_dump() if hasattr(structured_response, "model_dump") else structured_response,
            messages=messages
        )
    except Exception as e:
        # ---------------------------------------------------------------------
        # [LOGGING] Sub-Agent Execution Failed
        # ---------------------------------------------------------------------
        WorkflowLogger.log_subagent_failed(agent_name=agent_name, error=e)

        error_description = str(e)
        if error_correction_agent is not None:
            try:
                if hasattr(error_correction_agent, "start_fix"):
                    fix_res = error_correction_agent.start_fix(str(e))
                    if asyncio.iscoroutine(fix_res):
                        error_description = await fix_res
                    else:
                        error_description = str(fix_res)
                elif hasattr(error_correction_agent, "describe_error"):
                    error_description = await error_correction_agent.describe_error(str(e))
            except Exception as ec_err:
                WorkflowLogger.log_warning(f"Error correction agent failed on executor invoke error: {ec_err}")

        # Record failed sub-agent checkpoint
        if checkpoints is not None:
            agent_cp = AgentCheckpoint(
                agent=agent_name,
                status=f"Error: {str(e)}",
                task=agent_task,
                tools=[],
                output=f"Error executing task: {error_description}"
            )
            target_layer_cp = next((cp for cp in checkpoints if cp.layer_number == layer_number), None)
            if target_layer_cp is None:
                target_layer_cp = CheckpointDetails(layer_number=layer_number, agents=[])
                checkpoints.append(target_layer_cp)
            target_layer_cp.agents.append(agent_cp)

        return SubAgentExecutionResult(
            name=agent_name,
            task=agent_task,
            status="error",
            error=str(e),
            response=f"Error executing task: {error_description}",
            messages=[]
        )


async def layer_execution_engine(
    executable_layers: Optional[Union[ExecutableLayers, Dict[str, Any], List[Any]]] = None,
    initial_context: Optional[str] = None,
    checkpoints: Optional[List[CheckpointDetails]] = None,
    default_model_name: Optional[str] = None,
    error_correction_agent: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Layer Execution Engine Node: Executes the agent layers sequentially using the ExecutableLayers Pydantic schema.
    Within each layer, all sub-agents run in parallel using asyncio.gather.
    Outputs from each completed layer are accumulated and fed into all sub-agents in the subsequent layer as context.
    
    Args:
        executable_layers (Optional[Union[ExecutableLayers, Dict[str, Any], List[Any]]]): The ExecutableLayers Pydantic model, dictionary, or list.
        initial_context (Optional[str]): Optional initial context to pass to the first layer.
        checkpoints (Optional[List[CheckpointDetails]]): The list of checkpoint details for tracking layer execution.
        default_model_name (Optional[str]): Fallback model name for sub-agents if not specified.
        error_correction_agent (Optional[Any]): Sub-agent for error diagnosis and recovery.
        
    Returns:
        Dict[str, Any]: Detailed execution summary conforming to ExecutionSummary containing layer-by-layer results, accumulated context, and the final response.
    """
    # 1. Resolve executable layers container
    layers_container = executable_layers
    if layers_container is None:
        # ---------------------------------------------------------------------
        # [LOGGING] Warning: No executable layers
        # ---------------------------------------------------------------------
        WorkflowLogger.log_warning("No executable layers available to run.")
        return ExecutionSummary(
            execution_history=[],
            accumulated_context="",
            final_response="No layers were executed.",
            total_layers_executed=0
        ).model_dump()

    if isinstance(layers_container, ExecutableLayers):
        layers_list = layers_container.layers
    elif isinstance(layers_container, dict):
        layers_list = layers_container.get("layers", [])
    elif isinstance(layers_container, list):
        layers_list = layers_container
    else:
        layers_list = getattr(layers_container, "layers", [])

    if not layers_list:
        # ---------------------------------------------------------------------
        # [LOGGING] Warning: Empty layers list
        # ---------------------------------------------------------------------
        WorkflowLogger.log_warning("No executable layers available to run.")
        return ExecutionSummary(
            execution_history=[],
            accumulated_context="",
            final_response="No layers were executed.",
            total_layers_executed=0
        ).model_dump()

    # Sort layers by layer_number in ascending order (1, 2, 3...)
    layers_list = sorted(
        layers_list,
        key=lambda l: int(l.layer_number if isinstance(l, ExecutableLayer) else (l.get("layer_number") or 0 if isinstance(l, dict) else getattr(l, "layer_number", 0))),
        reverse=False
    )

    execution_history: List[LayerExecutionResult] = []
    accumulated_context = initial_context or ""

    # 2. Sequential layer-by-layer execution
    for layer in layers_list:
        if isinstance(layer, ExecutableLayer):
            layer_num = layer.layer_number
            sub_agents = layer.sub_agents
        elif isinstance(layer, dict):
            layer_num = layer.get("layer_number", len(execution_history) + 1)
            sub_agents = layer.get("sub_agents", [])
        else:
            layer_num = getattr(layer, "layer_number", len(execution_history) + 1)
            sub_agents = getattr(layer, "sub_agents", [])

        # ---------------------------------------------------------------------
        # [LOGGING] Layer Execution Starting
        # ---------------------------------------------------------------------
        WorkflowLogger.log_layer_start(layer_num=layer_num, sub_agent_count=len(sub_agents))

        # Run all sub-agents in this layer in parallel
        layer_results: List[SubAgentExecutionResult] = await asyncio.gather(
            *[
                run_single_agent(
                    agent=agent,
                    checkpoints=checkpoints,
                    layer_number=layer_num,
                    default_model_name=default_model_name,
                    error_correction_agent=error_correction_agent
                )
                for agent in sub_agents
            ]
        )

        # Format and accumulate this layer's output to feed into next layer
        layer_summary_parts = []
        for agent_res in layer_results:
            layer_summary_parts.append(
                f"Agent: {agent_res.name}\n"
                f"Task: {agent_res.task}\n"
                f"Output: {agent_res.response}"
            )

        this_layer_summary = f"[Layer {layer_num} Outputs]\n" + "\n\n".join(layer_summary_parts)

        if accumulated_context:
            accumulated_context += f"\n\n{this_layer_summary}"
        else:
            accumulated_context = this_layer_summary

        execution_history.append(LayerExecutionResult(
            layer_number=layer_num,
            sub_agent_results=layer_results,
            layer_summary=this_layer_summary
        ))

        # ---------------------------------------------------------------------
        # [LOGGING] Layer Execution Completed
        # ---------------------------------------------------------------------
        WorkflowLogger.log_layer_complete(layer_num=layer_num, result_count=len(layer_results))

    # -------------------------------------------------------------------------
    # [LOGGING] All Layers Completed Successfully
    # -------------------------------------------------------------------------
    WorkflowLogger.log_all_layers_complete(total_layers=len(execution_history))

    # 3. Determine final response and structured output from the last layer's sub-agents
    last_layer_agents = execution_history[-1].sub_agent_results if execution_history else []
    final_agent = last_layer_agents[-1] if last_layer_agents else None
    final_response = (
        final_agent.response
        if final_agent
        else ""
    )
    final_structured_output = (
        final_agent.structured_response
        if final_agent
        else None
    )

    # -------------------------------------------------------------------------
    # [LOGGING] Final Synthesized Response & Structured Output
    # -------------------------------------------------------------------------
    WorkflowLogger.log_final_response(final_response, structured_output=final_structured_output)

    # 4. Construct final execution summary
    summary = ExecutionSummary(
        execution_history=execution_history,
        accumulated_context=accumulated_context,
        final_response=final_response,
        structured_output=final_structured_output,
        total_layers_executed=len(execution_history)
    )

    return summary.model_dump()
