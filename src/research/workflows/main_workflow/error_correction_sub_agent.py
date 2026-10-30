"""
Error Correction Sub-Agent for Main Workflow.

This agent analyzes failures or errors encountered by sub-agents.
It creates an agent using langchain.agents.create_agent and prints
only the description of the error.
"""

from typing import Optional, List, Any
from langchain.agents import create_agent
from langchain_openrouter import ChatOpenRouter
from research.config import settings


def create_error_correction_agent(
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.2,
    tools: Optional[List[Any]] = None,
    name: str = "ErrorCorrectionAgent"
):
    """
    Creates an Error Correction Agent in a simple way using langchain.agents.create_agent.
    
    Args:
        model_name: Model name for LLM (defaults to settings.MAIN_WORKFLOW_MODEL).
        api_key: OpenRouter API key (defaults to settings.OPENROUTER_API_KEY).
        temperature: Model temperature (defaults to 0.2).
        tools: Optional list of tools.
        name: Name of the agent.
        
    Returns:
        Compiled agent executor.
    """
    llm = ChatOpenRouter(
        model=model_name or settings.MAIN_WORKFLOW_MODEL,
        temperature=temperature,
        api_key=api_key or settings.OPENROUTER_API_KEY
    )

    system_prompt = (
        "You are an AI Error Correction Agent. Your task is to analyze the failure or error "
        "encountered by a sub-agent and provide a clear, concise description of the error, "
        "identifying what caused the failure and how to resolve it."
    )

    return create_agent(
        model=llm,
        tools=tools or [],
        system_prompt=system_prompt,
        name=name
    )


class ErrorCorrectionSubAgent:
    """
    Sub-agent responsible for analyzing errors and displaying only the description of the error.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.2
    ):
        self.api_key = api_key or settings.OPENROUTER_API_KEY
        self.model_name = model_name or settings.MAIN_WORKFLOW_MODEL
        self.temperature = temperature

        # Create agent in a simple way
        self.agent = create_error_correction_agent(
            model_name=self.model_name,
            api_key=self.api_key,
            temperature=self.temperature
        )

    async def describe_error(
        self,
        error_message: str,
        agent_name: str = "SubAgent",
        task: str = "",
        tools_available: Optional[List[str]] = None,
        accumulated_context: Optional[str] = None
    ) -> str:
        """
        Analyzes the error and prints only the description of the error.
        """
        user_message = (
            f"Sub-Agent: {agent_name}\n"
            f"Assigned Task: {task}\n"
            f"Error Encountered: {error_message}\n"
        )
        if tools_available:
            user_message += f"Available Tools: {', '.join(tools_available)}\n"
        if accumulated_context:
            user_message += f"Context from Previous Layers: {accumulated_context}\n"

        user_message += "\nPlease analyze this error and provide only a clear description of the error."

        res = await self.agent.ainvoke({"messages": [("user", user_message)]})
        messages = res.get("messages", []) if isinstance(res, dict) else []
        error_description = str(messages[-1].content) if messages else str(res)

        # Show description of error only
        print(error_description)

        return error_description

    async def diagnose_error(
        self,
        error_message: str,
        agent_name: str = "SubAgent",
        task: str = "",
        tools_available: Optional[List[str]] = None,
        accumulated_context: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """
        Alias for describe_error for backwards compatibility.
        """
        return await self.describe_error(
            error_message=error_message,
            agent_name=agent_name,
            task=task,
            tools_available=tools_available,
            accumulated_context=accumulated_context
        )
