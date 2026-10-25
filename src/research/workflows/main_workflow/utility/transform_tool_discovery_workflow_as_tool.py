from typing import Optional, Any
from langchain_core.tools import StructuredTool
from research.workflows.tool_descovery_workflow.tool_discovery import ToolDiscoveryWorkflow


def transform_tool_discovery_workflow_as_tool(
    workflow: Optional[Any] = None
) -> Optional[StructuredTool]:
    """
    Transforms the tool_discovery_workflow into an executable LangChain StructuredTool
    that executes the discovery workflow and returns its complete workflow state.
    """
    target_wf = getattr(workflow, "tool_discovery_workflow", workflow)
    if not target_wf:
        return None

    async def _run_tool_discovery(task: str) -> dict:
        return await target_wf.tool_discovery_workflow_run(task)

    return StructuredTool.from_function(
        name="tool_discovery_workflow",
        description=(
            "Discovers, searches, tests, and evaluates API tools for a given task or capability. "
            "Call this tool if you need external APIs or tools to accomplish a task. "
            "Returns the discovered tool details (tool_title, description, argument, api_url, http_method, headers)."
        ),
        coroutine=_run_tool_discovery,
    )

