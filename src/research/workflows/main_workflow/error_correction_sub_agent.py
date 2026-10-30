"""
Error Correction Sub-Agent for Main Workflow.

This agent handles error correction for sub-agents.
Currently exposes start_fix to receive and print the error code.
"""

from typing import Optional, List, Any
from langchain.agents import create_agent
from langchain_openrouter import ChatOpenRouter


def create_error_correction_agent(
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.2,
    tools: Optional[List[Any]] = None,
    name: str = "ErrorCorrectionAgent"
):
    """
    Creates an Error Correction Agent using langchain.agents.create_agent.
    """
    llm = ChatOpenRouter(
        model=model_name or "deepseek/deepseek-v4-flash",
        temperature=temperature,
        api_key=api_key
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
        self.agent = create_error_correction_agent(
            model_name=self.model_name,
            api_key=self.api_key,
            temperature=self.temperature
        )

    async def start_fix(self, error_code: str) -> str:
        """
        Receives error_code only and prints error_code only for now.

        Args:
            error_code (str): The error code or message to fix.

        Returns:
            str: The error code.
        """
        print(error_code)
        return str(error_code)

   
