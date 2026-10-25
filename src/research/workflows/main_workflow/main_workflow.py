import json
import asyncio
from typing import List, Dict, Any, Union, Optional
from research.workflows.main_workflow.schemas import (
    NonExecutableSubAgentConfig,
    NonExecutableLayerConfig,
    OrchestratorOutput,
    ExecutableSubAgent,
    ExecutableLayer,
    ExecutableLayers,
    SubAgentExecutionResult,
    LayerExecutionResult,
    ExecutionSummary,
    AgentCheckpoint,
    CheckpointDetails,
)
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_openrouter import ChatOpenRouter
from langchain_core.tools import StructuredTool
from research.config import settings
from research.workflows.tool_descovery_workflow.tool_genarator import create_dynamic_api_tool
from research.workflows.tool_descovery_workflow.tool_discovery import ToolDiscoveryWorkflow, APIToolDefinition
from research.workflows.main_workflow.utility import (
    get_recent_models,
    get_each_model_details,
    get_design_arena_benchmarks,
    ai_tools_data,
    transform_tool_discovery_workflow_as_tool,
    build_layers_with_attaching_executable_agents,
    layer_execution_engine,
)
from research.workflows.main_workflow.workflow_logger import WorkflowLogger




class MainWorkFlow:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.5,
        tool_discovery_workflow: Optional[ToolDiscoveryWorkflow] = None
    ):
        self.api_key = api_key or settings.OPENROUTER_API_KEY
        self.model_name = model_name or settings.MAIN_WORKFLOW_MODEL
        self.temperature = temperature if temperature is not None else settings.MAIN_WORKFLOW_TEMPERATURE
        self.tool_discovery_workflow = tool_discovery_workflow
        
        # 1. Create tools from the db/data array
        # self.tools = [create_dynamic_api_tool(tool_dict) for tool_dict in ai_tools_data]
        
        # 2. Convert tool discovery workflow to executable tool returning workflow state and store in self (separate tool)
        self.tool_discovery_tool: Optional[StructuredTool] = transform_tool_discovery_workflow_as_tool(self.tool_discovery_workflow)
        
        # 3. Initialize LLM
        # self.llm = ChatOpenRouter(
        #     model=self.model_name,
        #     temperature=self.temperature,
        #     api_key=self.api_key
        # )
        
        self.mainAgent = ChatOpenRouter(
            model=self.model_name,
            temperature=self.temperature,
            api_key=self.api_key
        )

        # 4. Orchestrator Prompt
        self.orchestrator_prompt = (
            "You are a Task Orchestrator Agent. Your job is to divide the user's task into a sequential, multi-layer agentic workflow.\n"
            "Design the workflow layer by layer: all sub-agents in a single layer execute their tasks in parallel. "
            "Once that layer finishes, the next layer executes.\n\n"
            "TOOL DISCOVERY & CAPABILITIES:\n"
            "- You have access to the 'tool_discovery_workflow' tool.\n"
            "- CRITICAL RULE: Sub-agents CANNOT discover tools. If the user task requires external tools (e.g. weather, geocoding, crypto prices, search, etc.), YOU (the orchestrator) must call the 'tool_discovery_workflow' tool YOURSELF right now during planning.\n"
            "- NEVER delegate tool discovery to any sub-agent.\n"
            "- NEVER assign 'tool_discovery_workflow' to any sub-agent in their 'tools' list.\n"
            "- Once tools are discovered, assign ONLY the specific discovered APIToolDefinition objects to the sub-agents that need them.\n\n"
            "SUB-AGENT SPECIFICATIONS:\n"
            "- For each sub-agent, specify its name, system_prompt, task, model_name, and tools.\n"
            "- CRITICAL REQUIREMENT 1: You MUST strictly set the model_name field to 'deepseek/deepseek-v4-flash' for every sub-agent.\n"
            "- CRITICAL REQUIREMENT 2: In each sub-agent's 'tools' field, assign ONLY matching APIToolDefinition objects from discovered tools. If a sub-agent does not need any tools, set its 'tools' field to an empty list [].\n"
            "- CRITICAL REQUIREMENT 3: Absolutely NEVER include 'tool_discovery_workflow' in any sub-agent's 'tools' list. It is an orchestrator-only tool."
        )

        # 5. Initialize orchestrator memory store
        self.main_orchestrator_memory_store: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": self.orchestrator_prompt
            }
        ]

        # 6. Create main orchestrator agent connected directly with tool_discovery_tool
        self.orchestrator_agent = create_react_agent(
            model=self.mainAgent,
            tools=[self.tool_discovery_tool] if self.tool_discovery_tool else [],
            prompt=self.orchestrator_prompt,
            response_format=OrchestratorOutput,
            name="MainOrchestratorAgent"
        )

        # 7. Create agent & layers container
        self.executable_layers: Optional[ExecutableLayers] = None
        # self.agent_executor = create_react_agent(self.llm, tools=self.tools)

    # async def test_tool_agent(self, prompt: str) -> dict:
    #     """
    #     Method to test the main workflow tool agent.
        
    #     Args:-
    #         prompt (str): The prompt to test the tool agent with.
    #     """
    #     result = await self.agent_executor.ainvoke({"messages": [("user", prompt)]})
        
    #     return {
    #         "response": result["messages"][-1].content if result and "messages" in result else "",
    #         "tools_available": [t.name for t in self.tools]
    #     }

    async def main_orchestrator(self, state: Union[Dict[str, Any], str]) -> dict:
        """
        Main Orchestrator Node.
        Analyzes the user's main task and divides it into a multi-layered agentic workflow (layers executed sequentially, sub-agents within each layer executed in parallel),
        returning structured layer configurations adhering to OrchestratorOutput.
        The orchestrator is connected with tool_discovery_workflow as a tool; if needed, it calls the tool autonomously to discover tools and assigns them to sub-agents.
        
        Args:
            state (Union[Dict[str, Any], str]): The workflow state dict containing 'task' or the raw task string.
        """
        task = state if isinstance(state, str) else state.get("task", "")

        # -------------------------------------------------------------------------
        # [LOGGING] Workflow Planning Started
        # -------------------------------------------------------------------------
        WorkflowLogger.log_orchestrator_start(task)

        # 1. Append task to main_orchestrator_memory_store
        user_message = f"User Task: {task}"
        self.main_orchestrator_memory_store.append({
            "role": "user",
            "content": user_message
        })

        # 2. Invoke orchestrator agent (with tool discovery capability)
        orchestrator_res = await self.orchestrator_agent.ainvoke({
            "messages": [("user", user_message)]
        })

        # Extract structured response adhering to OrchestratorOutput
        structured_response = orchestrator_res.get("structured_response")
        if isinstance(structured_response, OrchestratorOutput):
            result = structured_response.model_dump()
        elif isinstance(structured_response, dict):
            result = structured_response
        else:
            # Fallback if structured_response is not directly present
            structured_agent = self.mainAgent.with_structured_output(OrchestratorOutput)
            raw_result = await structured_agent.ainvoke(orchestrator_res.get("messages", []))
            result = raw_result.model_dump() if hasattr(raw_result, "model_dump") else raw_result

        # -------------------------------------------------------------------------
        # [LOGGING] Log Orchestrator Result & Discovered Tool Calls
        # -------------------------------------------------------------------------
        WorkflowLogger.log_orchestrator_result(task=task, result=result, orchestrator_res=orchestrator_res)

        # 3. Append response to memory store
        self.main_orchestrator_memory_store.append({
            "role": "assistant",
            "content": json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else str(result)
        })

        # 4. Build executable agent layers
        layers_data = result.get("layers", []) if isinstance(result, dict) else getattr(result, "layers", [])
        self.executable_layers = build_layers_with_attaching_executable_agents(
            layers=layers_data,
            temperature=self.temperature,
            api_key=self.api_key,
            default_model_name=self.model_name
        )

        # 5. Initialize checkpoint variable before layer execution
        checkpoints: Optional[List[CheckpointDetails]] = []

        # 6. Execute layers sequentially with parallel sub-agents
        execution_result = await layer_execution_engine(
            executable_layers=self.executable_layers,
            checkpoints=checkpoints,
            default_model_name=self.model_name
        )

        # Reset / nullify checkpoints after execution run
        checkpoints = None

        return {
            "task_division": result,
            "execution_result": execution_result
        }