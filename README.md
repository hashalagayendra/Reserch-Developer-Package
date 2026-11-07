# Self Evolving Workflow (`self-evolving-workflow`)

Autonomous AI Tool Discovery and Research Agent Workflow Library.

## Features

- **Dynamic Tool Discovery**: Autonomous agent that searches a vector database or the web to locate or generate executable API tools for complex tasks.
- **Hierarchical Layered Agent Execution**: Multi-agent orchestration layer that decomposes complex problems into parallel sub-agents across execution stages.
- **Decoupled Architecture**: Can run against any compatible Tool Database API or standalone.

## Installation

### Local / Development Mode
In your backend application's environment:
```bash
# If using uv:
uv pip install -e ../research

# Or using standard pip:
pip install -e ../research
```

### Production / From PyPI (When Published)
```bash
pip install self-evolving-workflow
```

## Quick Start

```python
from research import Research

# Initialize and configure the research agent
researcher = Research()
researcher.configure(
    openRouterAPIKey="your-openrouter-key",
    toolDiscoveryModel="deepseek/deepseek-v4-flash",
    toolDiscoveryModelTemperature=0.8,
    toolDbBaseUrl="http://127.0.0.1:8000/api/v1",
    projectApiKey="your-project-api-key"
)
```

## MCP URL supplied by your backend

Pass only the separate MCP server's URL to the Research object. The package
discovers and converts its tools internally before creating the orchestrator:

```python
from research import Research


async def run_research(query, openrouter_key):
    researcher = Research()
    researcher.configure(
        openRouterAPIKey=openrouter_key,
        mcp_server_url="http://127.0.0.1:8001/mcp",
    )
    return await researcher.run(query)
```

The MCP adapter is a dependency of this package. `configure()` stores the URL;
`run()` automatically awaits `initialize()` to load `researcher.all_mcp_tools`
and build the main agent. To prepare the agent without running a query, call
`await researcher.initialize()` after configuring it. Initialization is reused
on subsequent runs; calling `configure()` resets it.

`researcher.all_mcp_tools` is copied into
`researcher.main_workflow.all_mcp_tools` as a callable registry. The main orchestrator
calls `extract_mcp_tools` to read names, descriptions, and input schemas, then
assigns selected names in each sub-agent configuration's separate `mcp_tool_names`
list. It cannot execute the remote MCP tools itself. API definitions discovered
through `tool_discovery_workflow` stay in the sub-agent's `tools` list.
The layer builder resolves MCP names against the registry; `create_single_agent`
combines these callable MCP tools with converted API tools in one agent executor.
Unknown MCP names and duplicate tool names are rejected before execution.
Omit `mcp_server_url` to preserve the original workflow. Tool names must be unique.

## Environment Variables

You can configure defaults via a `.env` file:
```ini
OPENROUTER_API_KEY=your_key_here
PROJECT_API_KEY=your_project_key
TOOL_DB_BASE_URL=http://127.0.0.1:8000/api/v1
TOOL_DISCOVERY_MODEL=deepseek/deepseek-v4-flash
TOOL_DISCOVERY_TEMPERATURE=0.8
MAIN_WORKFLOW_MODEL=deepseek/deepseek-v4-flash
MAIN_WORKFLOW_TEMPERATURE=0.0
EMBEDDING_MODEL=voyageai/voyage-4-lite
```

## Publishing to PyPI

When ready to publish:
```bash
# 1. Build the distribution package
uv build
# or: python -m build

# 2. Upload to PyPI
twine upload dist/*
```
