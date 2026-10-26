from typing import List, Dict, Any, Union, Optional
from research.config import settings
from research.workflows.main_workflow.schemas import (
    NonExecutableSubAgentConfig,
    NonExecutableLayerConfig,
    OrchestratorOutput,
    ExecutableSubAgent,
    ExecutableLayer,
    ExecutableLayers,
)
from research.workflows.main_workflow.workflow_logger import WorkflowLogger
from research.workflows.main_workflow.utility.create_single_agent import create_single_agent


def build_layers_with_attaching_executable_agents(
    layers: Union[List[Any], Dict[str, Any], OrchestratorOutput],
    tools: Optional[List[Any]] = None,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
    default_model_name: Optional[str] = None,
    response_schema: Optional[Any] = None
) -> ExecutableLayers:
    """
    Takes only the necessary layers data, iterates through each layer and sub-agent,
    sends the sub-agent data to create_single_agent to instantiate the executable agent,
    binds the response_schema to the final sub-agent of the last layer,
    and returns the layer-by-layer ExecutableLayers Pydantic schema containing the executable agents.
    
    Args:
        layers (Union[List[Any], Dict[str, Any], OrchestratorOutput]): The list of layers or task division data.
        tools (Optional[List[Any]], optional): Tools to bind to sub-agents.
        temperature (Optional[float], optional): Model temperature. Defaults to settings.MAIN_WORKFLOW_TEMPERATURE.
        api_key (Optional[str], optional): OpenRouter API key. Defaults to settings.OPENROUTER_API_KEY.
        default_model_name (Optional[str], optional): Default LLM model name if not specified in sub-agent config.
        response_schema (Optional[Any], optional): Structured response schema to bind to the last sub-agent of the last layer.
        
    Returns:
        ExecutableLayers: Layer-by-layer Pydantic model maintaining layer order and containing sub-agents with their executable agents.
    """
    resolved_api_key = api_key or settings.OPENROUTER_API_KEY
    resolved_temperature = temperature if temperature is not None else settings.MAIN_WORKFLOW_TEMPERATURE
    fallback_model = default_model_name or settings.MAIN_WORKFLOW_MODEL

    # 1. Extract raw layers list
    if isinstance(layers, OrchestratorOutput):
        raw_layers = layers.layers
    elif isinstance(layers, dict):
        raw_layers = layers.get("layers", [])
    elif isinstance(layers, list):
        raw_layers = layers
    else:
        raw_layers = []

    executable_layers_list: List[ExecutableLayer] = []
    total_layers = len(raw_layers)

    # 2. Build executable agents layer by layer
    for layer_idx, layer in enumerate(raw_layers):
        if isinstance(layer, NonExecutableLayerConfig):
            layer_num = layer.layer_number
            sub_agents_list = layer.sub_agents
        elif isinstance(layer, dict):
            layer_num = layer.get("layer_number", len(executable_layers_list) + 1)
            sub_agents_list = layer.get("sub_agents", [])
        else:
            layer_num = getattr(layer, "layer_number", len(executable_layers_list) + 1)
            sub_agents_list = getattr(layer, "sub_agents", [])

        layer_executable_sub_agents: List[ExecutableSubAgent] = []
        total_sub_agents = len(sub_agents_list)

        # 3. Iterate through each sub-agent in this layer and create its executable agent
        for agent_idx, agent_data in enumerate(sub_agents_list):
            if isinstance(agent_data, NonExecutableSubAgentConfig):
                name = agent_data.name
                system_prompt = agent_data.system_prompt
                task = agent_data.task
                model_name = agent_data.model_name
                agent_tools = agent_data.tools
            elif isinstance(agent_data, dict):
                name = agent_data.get("name", "SubAgent")
                system_prompt = agent_data.get("system_prompt", "You are a specialized sub-agent.")
                task = agent_data.get("task", "")
                model_name = agent_data.get("model_name") or fallback_model
                agent_tools = agent_data.get("tools", [])
            else:
                name = getattr(agent_data, "name", "SubAgent")
                system_prompt = getattr(agent_data, "system_prompt", "You are a specialized sub-agent.")
                task = getattr(agent_data, "task", "")
                model_name = getattr(agent_data, "model_name", fallback_model)
                agent_tools = getattr(agent_data, "tools", [])

            # Check if this agent is the very last sub-agent in the final layer
            is_last_agent = (layer_idx == total_layers - 1) and (agent_idx == total_sub_agents - 1)
            agent_response_format = response_schema if is_last_agent else None

            # Send data directly to create_single_agent utility function with only specifically assigned tools
            built_agent = create_single_agent(
                model_name=model_name,
                system_prompt=system_prompt,
                task=task,
                name=name,
                tools=agent_tools or [],
                temperature=resolved_temperature,
                api_key=resolved_api_key,
                response_format=agent_response_format
            )
            layer_executable_sub_agents.append(built_agent)

        executable_layers_list.append(ExecutableLayer(
            layer_number=layer_num,
            sub_agents=layer_executable_sub_agents
        ))

    # -------------------------------------------------------------------------
    # [LOGGING] Log generated executable layers tree
    # -------------------------------------------------------------------------
    WorkflowLogger.log_executable_layers(executable_layers_list)

    return ExecutableLayers(
        layers=executable_layers_list
    )
