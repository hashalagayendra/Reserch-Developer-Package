from typing import List, Any, Optional
from langchain.agents import create_agent
from langchain_openrouter import ChatOpenRouter
from research.config import settings
from research.workflows.tool_descovery_workflow.tool_genarator import create_dynamic_api_tool
from research.workflows.tool_descovery_workflow.tool_discovery import APIToolDefinition
from research.workflows.main_workflow.schemas import ExecutableSubAgent


def create_single_agent(
    model_name: str,
    system_prompt: str,
    task: str,
    name: str = "SubAgent",
    tools: Optional[List[Any]] = None,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
    response_format: Optional[Any] = None
) -> ExecutableSubAgent:
    """
    Creates an executable LangChain/LangGraph agent by sending model_name, system_prompt, and task.
    Dynamically generates executable tools using create_dynamic_api_tool for each APIToolDefinition or tool dictionary.
    
    Args:
        model_name (str): The LLM model name (e.g. 'deepseek/deepseek-v4-flash').
        system_prompt (str): The system prompt defining the persona and instructions.
        task (str): The specific task assigned to this agent.
        name (str, optional): Name of the agent. Defaults to "SubAgent".
        tools (Optional[List[Any]], optional): Tools or APIToolDefinition items to bind to this agent. Defaults to None (empty list).
        temperature (Optional[float], optional): Model temperature. Defaults to settings.MAIN_WORKFLOW_TEMPERATURE.
        api_key (Optional[str], optional): OpenRouter API key. Defaults to settings.OPENROUTER_API_KEY.
        response_format (Optional[Any], optional): Structured response schema (Pydantic model or schema) to bind to this agent.
        
    Returns:
        ExecutableSubAgent: Pydantic model instance containing agent details, initialized LLM model instance, and compiled executable agent executor.
    """
    agent_tools = tools if tools is not None else []
    agent_temp = temperature if temperature is not None else settings.MAIN_WORKFLOW_TEMPERATURE
    agent_api_key = api_key or settings.OPENROUTER_API_KEY

    # 1. Convert APIToolDefinition instances or tool dictionaries into executable tools using create_dynamic_api_tool
    executable_agent_tools: List[Any] = []
    for t in agent_tools:
        if isinstance(t, APIToolDefinition):
            executable_agent_tools.append(create_dynamic_api_tool(t.model_dump()))
        elif isinstance(t, dict):
            executable_agent_tools.append(create_dynamic_api_tool(t))
        else:
            executable_agent_tools.append(t)

    # 2. Initialize dedicated LLM model instance
    sub_agent_llm = ChatOpenRouter(
        model=model_name,
        temperature=agent_temp,
        api_key=agent_api_key
    )

    # 3. Build structured prompt combining system prompt and assigned task
    formatted_prompt = (
        "You are an AI Sub-Agent with the following instructions and role:\n"
        f"{system_prompt.strip()}\n\n"
        "================== ASSIGNED TASK ==================\n"
        f"{task.strip()}\n"
        "===================================================\n\n"
        "Execute the assigned task thoroughly using the available tools when needed."
    )

    if response_format is not None:
        formatted_prompt += (
            "\n\nCRITICAL REQUIREMENT: You MUST provide your final answer strictly adhering to "
            "the required structured response schema."
        )

    # 4. Generate executable Tool-Calling agent with dynamic API tools, system prompt, and optional response_format
    agent_executor = create_agent(
        model=sub_agent_llm,
        tools=executable_agent_tools,
        system_prompt=formatted_prompt,
        response_format=response_format,
        name=name
    )

    return ExecutableSubAgent(
        name=name,
        system_prompt=system_prompt,
        task=task,
        model_name=model_name,
        llm=sub_agent_llm,
        tools=executable_agent_tools,
        executor=agent_executor,
        response_format=response_format
    )
