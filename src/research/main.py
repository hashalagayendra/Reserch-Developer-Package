from research.workflows.tool_descovery_workflow.tool_discovery import ToolDiscoveryWorkflow
from research.workflows.main_workflow.main_workflow import MainWorkFlow
from research.db_operations.main import ToolDBClient
from research.config import settings
from typing import Optional, Type, Any, Generic, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T")


class ResponseSchema(BaseModel, Generic[T]):
    """
    Standard structured response schema wrapping the workflow output.
    The 'response' attribute type connects with the configured schema if provided,
    or is a direct string type if no schema was configured.
    """
    response: T = Field(description="The final comprehensive response or structured research output.")


class Research:
    openRouterAPIKey: str
    toolDiscoveryModel: str
    toolDiscoveryModelTemperature: float
    mainWorkflowModel: str
    mainWorkflowModelTemperature: float
    projectApiKey: Optional[str]
    toolDbBaseUrl: str
    response_schema: Type[BaseModel]

    def __init__(self):
        self.response_schema = ResponseSchema[str]

    def configure(
        self,
        openRouterAPIKey: Optional[str] = None,
        toolDiscoveryModel: Optional[str] = None,
        toolDiscoveryModelTemperature: Optional[float] = None,
        projectApiKey: Optional[str] = None,
        toolDbBaseUrl: Optional[str] = None,
        mainWorkflowModel: Optional[str] = None,
        mainWorkflowModelTemperature: Optional[float] = None,
        response_schema: Optional[Type[BaseModel]] = None
    ):
        """
        Configure the Research workflow agent.

        Args:
            openRouterAPIKey (str, optional): The OpenRouter API key. Defaults to settings.OPENROUTER_API_KEY.
            toolDiscoveryModel (str, optional): The model to use for tool discovery. Defaults to settings.TOOL_DISCOVERY_MODEL.
            toolDiscoveryModelTemperature (float, optional): Temperature for tool discovery. Defaults to settings.TOOL_DISCOVERY_TEMPERATURE.
            projectApiKey (str, optional): Project API key for the Tool Database. Defaults to settings.PROJECT_API_KEY.
            toolDbBaseUrl (str, optional): Base URL for the Tool DB API. Defaults to settings.TOOL_DB_BASE_URL.
            mainWorkflowModel (str, optional): Model for the main workflow. Defaults to settings.MAIN_WORKFLOW_MODEL.
            mainWorkflowModelTemperature (float, optional): Temperature for the main workflow. Defaults to settings.MAIN_WORKFLOW_TEMPERATURE.
            response_schema (Type[BaseModel], optional): Pydantic model for structured output from the final agent. Wrapped inside ResponseSchema['response'] (defaults to str if None).
        """
        openRouterAPIKey = openRouterAPIKey or settings.OPENROUTER_API_KEY
        toolDiscoveryModel = toolDiscoveryModel or settings.TOOL_DISCOVERY_MODEL
        if toolDiscoveryModelTemperature is None:
            toolDiscoveryModelTemperature = settings.TOOL_DISCOVERY_TEMPERATURE
        mainWorkflowModel = mainWorkflowModel or settings.MAIN_WORKFLOW_MODEL
        if mainWorkflowModelTemperature is None:
            mainWorkflowModelTemperature = settings.MAIN_WORKFLOW_TEMPERATURE

        if not isinstance(openRouterAPIKey, str) or not openRouterAPIKey.strip():
            raise ValueError("openRouterAPIKey must be a valid, non-empty string.")
            
        if not isinstance(toolDiscoveryModel, str) or not toolDiscoveryModel.strip():
            raise ValueError("toolDiscoveryModel must be a valid, non-empty string.")
            
        if not isinstance(toolDiscoveryModelTemperature, (int, float)):
            raise TypeError("toolDiscoveryModelTemperature must be a numeric value.")
            
        if not (0.0 <= toolDiscoveryModelTemperature <= 2.0):       
            raise ValueError("toolDiscoveryModelTemperature must be between 0.0 and 2.0.")
            
        if not isinstance(mainWorkflowModel, str) or not mainWorkflowModel.strip():
            raise ValueError("mainWorkflowModel must be a valid, non-empty string.")
            
        if not isinstance(mainWorkflowModelTemperature, (int, float)):
            raise TypeError("mainWorkflowModelTemperature must be a numeric value.")
            
        if not (0.0 <= mainWorkflowModelTemperature <= 2.0):       
            raise ValueError("mainWorkflowModelTemperature must be between 0.0 and 2.0.")
            
        self.openRouterAPIKey = openRouterAPIKey.strip()
        self.toolDiscoveryModel = toolDiscoveryModel.strip()
        self.toolDiscoveryModelTemperature = float(toolDiscoveryModelTemperature)
        self.mainWorkflowModel = mainWorkflowModel.strip()
        self.mainWorkflowModelTemperature = float(mainWorkflowModelTemperature)
        
        if response_schema is not None:
            try:
                self.response_schema = ResponseSchema[response_schema]
            except Exception:
                self.response_schema = response_schema
        else:
            self.response_schema = ResponseSchema[str]
        
        resolved_project_key = projectApiKey or settings.PROJECT_API_KEY
        self.projectApiKey = resolved_project_key.strip() if resolved_project_key else None
        self.toolDbBaseUrl = (toolDbBaseUrl or settings.TOOL_DB_BASE_URL).rstrip('/')

        # Initialize the DB Client
        db_client = ToolDBClient(base_url=self.toolDbBaseUrl, api_key=self.projectApiKey or "")

        self.tool_discovery_workflow = ToolDiscoveryWorkflow(
            openRouterAPIKey=self.openRouterAPIKey,
            toolDiscoveryModel=self.toolDiscoveryModel,
            toolDiscoveryModelTemperature=self.toolDiscoveryModelTemperature,
            db_client=db_client
        )
        
        self.main_workflow = MainWorkFlow(
            tool_discovery_workflow = self.tool_discovery_workflow,
            api_key=self.openRouterAPIKey,
            model_name=self.mainWorkflowModel,
            temperature=self.mainWorkflowModelTemperature,
            response_schema=self.response_schema
        )

    async def run(self, query: str):
        """ 
        Execute the full research workflow for the given user query.

        Args:
            query (str): The research task or user question.

        Returns:
            dict: The workflow execution result containing task division, execution history, and structured_output (if configured).
        """
        response = await self.main_workflow.main_orchestrator(query)
        return response