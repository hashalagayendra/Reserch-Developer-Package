"""
Error Correction Sub-Agent for Main Workflow.

This agent handles error correction for sub-agents.
Currently exposes start_fix to receive and print error correction details.
"""

from typing import Optional, List, Any, Dict, Union
from langchain.agents import create_agent
from langchain_openrouter import ChatOpenRouter
from research.workflows.main_workflow.schemas import (
    ExecutableSubAgent,
    CheckpointDetails,
)


class ErrorCorrectionSubAgent:
    """
    Sub-agent responsible for error correction.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.2
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature

        # Create agent
        self.error_correction_agent = self.create_error_correction_agent()
       

    def create_error_correction_agent(
        self,
        tools: Optional[List[Any]] = None,
        name: str = "ErrorCorrectionAgent"
    ):
        """
        Creates an Error Correction Agent using langchain.agents.create_agent.
        Uses self attributes (self.model_name, self.temperature, self.api_key) directly.
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

        return create_agent(
            model=llm,
            tools=tools or [],
            system_prompt=system_prompt,
            name=name
        )

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

        Args:
            agent (Union[ExecutableSubAgent, Dict[str, Any]]): The failed sub-agent instance or config dictionary.
            checkpoints (Optional[List[CheckpointDetails]], optional): List of checkpoint details.
            layer_number (Optional[int], optional): Current execution layer number.
            default_model_name (Optional[str], optional): Fallback model name.
            error (Optional[Any], optional): The exception or error that occurred.

        Returns:
            Any: The error or fix result.
        """
        # Store input data in local variables
        agent_data = agent
        checkpoints_data = checkpoints
        layer_number_data = layer_number
        default_model_name_data = default_model_name
        error_data = error

        # Extract agent-specific details into local variables
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
