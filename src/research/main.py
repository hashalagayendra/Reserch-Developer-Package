from research.workflows.tool_descovery_workflow.tool_discovery import ToolDiscoveryWorkflow
from research.workflows.main_workflow.main_workflow import MainWorkFlow
from research.db_operations.main import ToolDBClient
from research.config import settings
from typing import Optional

class Research:
    openRouterAPIKey: str
    toolDiscoveryModel: str
    toolDiscoveryModelTemperature: float
    mainWorkflowModel: str
    mainWorkflowModelTemperature: float
    projectApiKey: Optional[str]
    toolDbBaseUrl: str

    def __init__(self):
        pass

    def configure(
        self,
        openRouterAPIKey: Optional[str] = None,
        toolDiscoveryModel: Optional[str] = None,
        toolDiscoveryModelTemperature: Optional[float] = None,
        projectApiKey: Optional[str] = None,
        toolDbBaseUrl: Optional[str] = None,
        mainWorkflowModel: Optional[str] = None,
        mainWorkflowModelTemperature: Optional[float] = None
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
            temperature=self.mainWorkflowModelTemperature
        )

    # async def run(self, query: str):
    #     """ 
    #     Args:-
    #         query (str): The query to run the tool discovery agent with.
    #     """
       
    #     response = await self.tool_discovery_workflow.tool_discovery_workflow_run(query)
       
    #     return response

