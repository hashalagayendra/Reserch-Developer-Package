import operator
import logging
from typing import TypedDict, Annotated, List, Dict, Any

from langchain_openrouter import ChatOpenRouter
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from research.config import settings
from research.workflows.main_workflow.workflow_logger import WorkflowLogger

# --- STRUCTURED OUTPUT MODELS ---
class TaskAnalysis(BaseModel):
    search_queries: List[str] = Field(
        description="A list of specific search queries to find the tools needed to complete the task."
    )

class ToolEvaluation(BaseModel):
    is_sufficient: bool = Field(
        description="True if the provided tools are completely sufficient to accomplish the task, False otherwise."
    )
    missing_capabilities: List[str] = Field(
        description="A list of specific tool capabilities that are still missing to accomplish the task. Empty if is_sufficient is True."
    )

class APIToolDefinition(BaseModel):
    tool_title: str = Field(description="A short, unique, camelCase or snake_case name for the tool.")
    description: str = Field(description="A clear description of what the tool does.")
    argument: Dict[str, Any] = Field(
        description="A dictionary of arguments. Each key is the argument name, and the value is a dictionary containing 'type' (e.g. 'string', 'number', 'boolean', 'integer'), 'description', and 'in' ('query', 'path', 'body', or 'header').",
        default_factory=dict
    )
    api_url: str = Field(description="The full URL for the API endpoint. If path variables are needed, use {param_name} placeholders in the URL.")
    http_method: str = Field(description="The HTTP method to use (e.g., 'GET', 'POST', 'PUT').", default="GET")
    headers: Dict[str, str] = Field(
        description="Any required HTTP headers, like {'Accept': 'application/json'} or {'Content-Type': 'application/json'}.",
        default_factory=dict
    )
    test_arguments: Dict[str, Any] = Field(
        description="A dictionary containing valid, realistic test values for each of the arguments defined above, to be used when testing the API.",
        default_factory=dict
    )

# --- STATE DEFINITION ---
class ToolDiscoveryState(TypedDict):
    task: str
    search_queries: List[str]
    local_tools_found: List[Dict]
    is_sufficient: bool
    missing_capabilities: List[str]
    web_tools_found: List[Dict]
    tested_tools: List[Dict]
    final_answer: str

# --- WORKFLOW CLASS ---
class ToolDiscoveryWorkflow:
    def __init__(
        self, 
        openRouterAPIKey: str = None, 
        toolDiscoveryModel: str = None, 
        toolDiscoveryModelTemperature: float = None,
        embeddingModel: str = None,
        db_client = None
    ):
        self.db_client = db_client
        # Configuration
        api_key = openRouterAPIKey or settings.OPENROUTER_API_KEY
        model = toolDiscoveryModel or settings.TOOL_DISCOVERY_MODEL
        temp = toolDiscoveryModelTemperature if toolDiscoveryModelTemperature is not None else settings.TOOL_DISCOVERY_TEMPERATURE
        emb_model = embeddingModel or settings.EMBEDDING_MODEL
        
        # Initialize the LLM
        self.llm = ChatOpenRouter(
            model=model,
            temperature=temp,
            api_key=api_key
        )
        
        # Initialize Embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=api_key,
            model=emb_model,
            check_embedding_ctx_length=False,
            tiktoken_enabled=False
        )

        # Generate the custom graph
        self.workflow = self.generate_tool_discovery_workflow()

    def generate_tool_discovery_workflow(self):
        """Builds the advanced Tool Discovery Graph."""
        workflow = StateGraph(ToolDiscoveryState)
        
        # Add Nodes
        workflow.add_node("analyze_task", self.node_analyze_task)
        workflow.add_node("explore_db", self.node_explore_db)
        workflow.add_node("evaluate_tools", self.node_evaluate_tools)
        workflow.add_node("web_search", self.node_web_search)
        workflow.add_node("test_tool", self.node_test_tool)
        workflow.add_node("store_tool", self.node_store_tool)
        workflow.add_node("final_answer", self.node_final_answer)

        # Connect Nodes (Edges)
        workflow.add_edge(START, "analyze_task")
        workflow.add_edge("analyze_task", "explore_db")
        workflow.add_edge("explore_db", "evaluate_tools")
        
        # Conditional Edge after Evaluation
        workflow.add_conditional_edges(
            "evaluate_tools",
            self.route_after_evaluate,
            {
                "final_answer": "final_answer",
                "web_search": "web_search"
            }
        )
        
        workflow.add_edge("web_search", "test_tool")
        workflow.add_edge("test_tool", "store_tool")
        workflow.add_edge("store_tool", "final_answer")
        workflow.add_edge("final_answer", END)
        
        return workflow.compile()

    # --- ROUTING LOGIC ---
    def route_after_evaluate(self, state: ToolDiscoveryState) -> str:
        """If local tools are sufficient, go to final answer. Else, search the web."""
        if state.get("is_sufficient"):
            return "final_answer"
        return "web_search"

    # --- NODES ---
    def node_analyze_task(self, state: ToolDiscoveryState):
        """Uses LLM to break down the complex task into specific search queries."""
        task = state.get("task", "")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI architect. Break down the user's task and determine what distinct, GENERIC AI tools or APIs are required to accomplish it. DO NOT include specific data (like a specific country, city, or name) in the search queries. Instead, extract the general capability needed (e.g., 'Global Weather API' instead of 'Sri Lanka Weather API'). Output a list of these generic search queries to find the necessary tools."),
            ("user", "Task: {task}")
        ])
        
        chain = prompt | self.llm.with_structured_output(TaskAnalysis)
        try:
            result = chain.invoke({"task": task})
            queries = result.search_queries
        except Exception as e:
            WorkflowLogger.log_warning(f"Failed to parse structured output in analyze_task: {e}. Using raw task as query.")
            queries = [task]
            
        # -------------------------------------------------------------------------
        # [LOGGING] Task Breakdown & Generated Search Queries
        # -------------------------------------------------------------------------
        WorkflowLogger.log_tool_discovery_analysis(task=task, queries=queries)
        return {"search_queries": queries}

    def node_explore_db(self, state: ToolDiscoveryState):
        """Searches the local PostgreSQL vector database for each query."""
        queries = state.get("search_queries", [])
        found_tools = []
        seen_titles = set()
        query_details = []
        
        for query in queries:
            query_vector = self.embeddings.embed_query(query)
            
            # Get top 2 closest matches per query using db_client
            if self.db_client:
                tools = self.db_client.search_tools(query_vector=query_vector, limit=2)
            else:
                tools = []
                
            query_details.append({
                "query": query,
                "vector_dim": len(query_vector),
                "matches": len(tools)
            })
            
            if tools:
                for t in tools:
                    tool_title = t.get("tool_title")
                    if tool_title not in seen_titles:
                        seen_titles.add(tool_title)
                        found_tools.append({
                            "tool_title": tool_title,
                            "description": t.get("description"),
                            "argument": t.get("argument"),
                            "api_url": t.get("api_url"),
                            "http_method": t.get("http_method"),
                            "headers": t.get("headers")
                        })
            
        # -------------------------------------------------------------------------
        # [LOGGING] Vector Database Search Matches & Query Details
        # -------------------------------------------------------------------------
        WorkflowLogger.log_tool_discovery_db_search(
            queries=queries,
            tools_found=found_tools,
            query_details=query_details
        )
        return {"local_tools_found": found_tools}

    def node_evaluate_tools(self, state: ToolDiscoveryState):
        """Uses LLM to evaluate if the found local tools are sufficient to complete the task."""
        task = state.get("task", "")
        local_tools = state.get("local_tools_found", [])
        
        tools_desc = "\n".join([f"- {t['tool_title']}: {t['description']}" for t in local_tools]) if local_tools else "No local tools found."
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a tool evaluator. You will be given a task and a list of available tools. Determine if the provided tools are completely sufficient to accomplish the task. If not, list the GENERIC missing capabilities needed. Avoid including specific data like locations or names in the capability description (e.g., ask for a 'General Weather API' instead of 'Tools to look up weather in Sri Lanka')."),
            ("user", "Task: {task}\n\nAvailable Tools:\n{tools}")
        ])
        
        chain = prompt | self.llm.with_structured_output(ToolEvaluation)
        try:
            result = chain.invoke({"task": task, "tools": tools_desc})
            is_sufficient = result.is_sufficient
            missing = result.missing_capabilities
        except Exception as e:
            WorkflowLogger.log_warning(f"Failed to parse structured output in evaluate_tools: {e}.")
            is_sufficient = False
            missing = ["Unknown missing capabilities"]
            
        # -------------------------------------------------------------------------
        # [LOGGING] Tool Sufficiency Evaluation
        # -------------------------------------------------------------------------
        WorkflowLogger.log_tool_discovery_evaluation(is_sufficient=is_sufficient, missing_capabilities=missing)
        return {"is_sufficient": is_sufficient, "missing_capabilities": missing}

    def node_web_search(self, state: ToolDiscoveryState):
        """Searches the web for specifically missing capabilities and extracts an API definition."""
        from duckduckgo_search import DDGS
        
        missing = state.get("missing_capabilities", [])
        web_tools = []
        for capability in missing:
            query = f"{capability} open free API without API key JSON"
            try:
                results = DDGS().text(query, max_results=10)
                search_text = "\n".join([f"Title: {r.get('title')}\nSnippet: {r.get('body')}\nURL: {r.get('href')}\n" for r in results])
                
                # -----------------------------------------------------------------
                # [LOGGING] DuckDuckGo Web Search Query & Raw Results Preview
                # -----------------------------------------------------------------
                WorkflowLogger.log_tool_discovery_web_search(
                    capability=capability,
                    query=query,
                    results_count=len(results),
                    preview_snippet=search_text[:600]
                )
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are an API extraction assistant. Your job is to extract a runnable, public API definition from the given search results that matches the user's capability requirement.\n"
                               "CRITICAL REQUIREMENTS:\n"
                               "1. The API MUST be completely free and MUST NOT require any API keys, authentication, or registration.\n"
                               "2. Do not hallucinate APIs. If all search results require an API key, fallback to a known open free API (like Open-Meteo for weather, CoinGecko for crypto, genderize.io, etc.).\n"
                               "3. Method & Parameter Structure:\n"
                               "   - For each parameter in `argument`, provide 'type', 'description', and 'in' ('query', 'path', 'body', or 'header').\n"
                               "   - For GET APIs: Provide clean base URL in `api_url` or use `{{param_name}}` in URL for path variables. Set `in: 'query'` for URL query params, or `in: 'path'` for path variables.\n"
                               "   - For POST/PUT APIs: Set `http_method` to 'POST', specify headers (e.g. {{'Content-Type': 'application/json'}}), and set `in: 'body'` for request body fields.\n"
                               "4. Provide realistic, working sample values in `test_arguments` matching every argument defined in `argument` so the tool can be executed and validated.\n"
                               "5. Always return a valid JSON API definition."),
                    ("user", "Capability required: {capability}\n\nSearch Results:\n{search_results}")
                ])
                
                chain = prompt | self.llm.with_structured_output(APIToolDefinition)
                api_def = chain.invoke({"capability": capability, "search_results": search_text})
                
                tool_dict = {
                    "tool_title": api_def.tool_title,
                    "description": api_def.description,
                    "argument": api_def.argument,
                    "api_url": api_def.api_url,
                    "http_method": api_def.http_method,
                    "headers": api_def.headers,
                    "test_arguments": api_def.test_arguments
                }
                
                # -----------------------------------------------------------------
                # [LOGGING] Extracted Free Public API Definition
                # -----------------------------------------------------------------
                WorkflowLogger.log_tool_discovery_web_extracted(
                    capability=capability,
                    tool_def=tool_dict
                )
                
                web_tools.append(tool_dict)
            except Exception as e:
                WorkflowLogger.log_warning(f"Failed to find or parse API for capability '{capability}': {e}")
                
        return {"web_tools_found": web_tools}

    def node_test_tool(self, state: ToolDiscoveryState):
        """Simulates testing the newly discovered web tools by creating a runnable tool and executing it."""
        from research.workflows.tool_descovery_workflow.tool_genarator import create_dynamic_api_tool
        
        web_tools = state.get("web_tools_found", [])
        tested = []
        for wt in web_tools:
            tool_title = wt.get("tool_title", "Unknown")
            mock_args = dict(wt.get("test_arguments") or {})
            arg_schema = wt.get("argument") or {}
            
            try:
                # 1. Create the actual runnable tool!
                runnable_tool = create_dynamic_api_tool(wt)
                
                # If the tool requires arguments but no test_arguments were generated, raise an exception
                if arg_schema and not mock_args:
                    raise ValueError(f"No test arguments generated for tool '{tool_title}' with argument schema: {list(arg_schema.keys())}")

                # 2. Execute the tool with test arguments
                test_result = runnable_tool.invoke(mock_args)
                
                if isinstance(test_result, str) and test_result.startswith("Error executing tool"):
                    raise Exception(test_result)
                
                wt["tested_and_passed"] = True
                # -----------------------------------------------------------------
                # [LOGGING] Discovered Tool Test Passed
                # -----------------------------------------------------------------
                WorkflowLogger.log_tool_discovery_test(
                    tool_title=tool_title,
                    test_args=mock_args,
                    passed=True,
                    result_preview=str(test_result)
                )
            except Exception as e:
                wt["tested_and_passed"] = False
                # -----------------------------------------------------------------
                # [LOGGING] Discovered Tool Test Failed
                # -----------------------------------------------------------------
                WorkflowLogger.log_tool_discovery_test(
                    tool_title=tool_title,
                    test_args=mock_args,
                    passed=False,
                    result_preview=str(e)
                )
                
            tested.append(wt)
            
        return {"tested_tools": tested}

    def node_store_tool(self, state: ToolDiscoveryState):
        """Stores the successfully tested tools into the local database."""
        tested_tools = state.get("tested_tools", [])
        
        if not self.db_client:
            WorkflowLogger.log_warning("[DB] No db_client provided, skipping DB store.")
            return {}
            
        stored_count = 0
        for tt in tested_tools:
            if tt.get("tested_and_passed"):
                try:
                    vector = self.embeddings.embed_query(tt["description"])
                    # Using hardcoded project_id=1 as a dummy for now
                    tool_data = {
                        "project_id": 1,
                        "tool_title": tt["tool_title"],
                        "description": tt["description"],
                        "argument": tt["argument"],
                        "api_url": tt.get("api_url"),
                        "http_method": tt.get("http_method", "GET"),
                        "headers": tt.get("headers", {}),
                        "vector_data": vector
                    }
                    self.db_client.store_tool(tool_data)
                    stored_count += 1
                except Exception as e:
                    WorkflowLogger.log_warning(f"Error storing tool '{tt.get('tool_title')}': {e}")
                    
        # -------------------------------------------------------------------------
        # [LOGGING] Caching Validated Tools to Vector Database
        # -------------------------------------------------------------------------
        WorkflowLogger.log_tool_discovery_stored(stored_count=stored_count)
        return {}

    def node_final_answer(self, state: ToolDiscoveryState):
        """Generates the final response based on what was found."""
        local = state.get("local_tools_found", [])
        new_tools = state.get("tested_tools", [])
        is_sufficient = state.get("is_sufficient")
        
        msg = f"Task Analysis Complete. Local Tools Sufficient: {is_sufficient}\n\n"
        
        if local:
            msg += "(From Local Database)\n"
            for t in local:
                msg += f"- {t['tool_title']}: {t['description']}\n"
        
        if new_tools:
            msg += "\n(Newly Discovered & Tested from Web)\n"
            for t in new_tools:
                msg += f"- {t['tool_title']}: {t['description']}\n"
                
        # -------------------------------------------------------------------------
        # [LOGGING] Tool Discovery Final Answer Synthesis
        # -------------------------------------------------------------------------
        WorkflowLogger.log_tool_discovery_final(final_answer=msg)
        return {"final_answer": msg}

    async def tool_discovery_workflow_run(self, task: str):
        """Helper method to run the workflow with a complex task."""
        # -------------------------------------------------------------------------
        # [LOGGING] Tool Discovery Workflow Started
        # -------------------------------------------------------------------------
        WorkflowLogger.log_tool_discovery_start(task=task)
        initial_state = {
            "task": task, 
            "search_queries": [],
            "local_tools_found": [], 
            "is_sufficient": False,
            "missing_capabilities": [],
            "web_tools_found": [],
            "tested_tools": [],
            "final_answer": ""
        }
        # Run the custom graph
        response = await self.workflow.ainvoke(initial_state)
        
        local_tools = response.get("local_tools_found") or []
        tested_tools = [t for t in (response.get("tested_tools") or []) if t.get("tested_and_passed")]
        all_ready = local_tools + tested_tools
        # -------------------------------------------------------------------------
        # [LOGGING] Tool Discovery Workflow Complete Summary
        # -------------------------------------------------------------------------
        WorkflowLogger.log_tool_discovery_complete(tools_found=all_ready)
        return response
