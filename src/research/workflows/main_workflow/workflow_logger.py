"""
Workflow Logger for Main Workflow & Tool Discovery.

Provides visual, rich-formatted terminal logging for:
- Orchestrator planning & tool discovery workflows
- Layer-by-layer sub-agent topology and bindings
- Dynamic API tool requests, responses, HTTP status, and latency (ms)
- Sub-agent execution steps, tool invocations, and raw output cards
- Layer execution rules, status badges, and final synthesized responses
Uses the 'rich' library to render boxes, trees, tables, syntax-highlighted JSON, and colored status badges.
"""

import sys
import json
import time
from typing import Any, Dict, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.syntax import Syntax
from rich import box

# Ensure Windows terminal handles UTF-8 emojis and box characters cleanly
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

console = Console()


class WorkflowLogger:
    """Centralized visual logger for the entire workflow execution lifecycle."""

    # =========================================================================
    # 1. ORCHESTRATOR PLANNING LOGS
    # =========================================================================

    @classmethod
    def log_orchestrator_start(cls, task: str) -> None:
        """Visual panel for the start of orchestrator planning."""
        console.print()
        console.print(
            Panel(
                f"[bold white]{task}[/bold white]",
                title="[bold yellow]🎯 TASK ORCHESTRATOR PLANNING[/bold yellow]",
                subtitle="[dim]Dividing task into sequential layers & parallel agents[/dim]",
                border_style="yellow",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )

    @classmethod
    def log_orchestrator_result(
        cls,
        task: str,
        result: Any,
        orchestrator_res: Optional[Dict[str, Any]] = None
    ) -> None:
        """Visual display of tool calls made, structured plan table, and raw JSON result."""
        # 1. Log any tool calls made by the orchestrator (e.g. tool_discovery_workflow)
        if orchestrator_res:
            messages = orchestrator_res.get("messages", [])
            tool_calls_made = []
            for msg in messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tc_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", str(tc))
                        tc_args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                        tool_calls_made.append((tc_name, tc_args))

            if tool_calls_made:
                table = Table(
                    title="🔧 Autonomous Tool Calls (Orchestrator)",
                    border_style="cyan",
                    box=box.ROUNDED,
                    show_header=True,
                    header_style="bold cyan"
                )
                table.add_column("Tool Name", style="bold yellow", width=30)
                table.add_column("Arguments (Raw)", style="white")
                for name, args in tool_calls_made:
                    arg_str = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
                    table.add_row(name, arg_str)
                console.print(table)
                console.print()

        # 2. Render structured Plan Table if 'layers' is present
        layers = []
        if isinstance(result, dict):
            layers = result.get("layers", [])
        elif hasattr(result, "layers"):
            layers = getattr(result, "layers", [])

        if layers:
            plan_table = Table(
                title=f"📋 Task Orchestration Plan: \"{task[:60]}\"",
                border_style="green",
                box=box.ROUNDED,
                show_lines=True,
                header_style="bold green"
            )
            plan_table.add_column("Layer", style="bold yellow", justify="center", width=8)
            plan_table.add_column("Sub-Agent", style="bold cyan", width=24)
            plan_table.add_column("Model", style="magenta", width=26)
            plan_table.add_column("Assigned Task", style="white")
            plan_table.add_column("Tools", style="bright_blue", width=20)

            for layer in layers:
                if isinstance(layer, dict):
                    layer_num = str(layer.get("layer_number", "?"))
                    sub_agents = layer.get("sub_agents", [])
                else:
                    layer_num = str(getattr(layer, "layer_number", "?"))
                    sub_agents = getattr(layer, "sub_agents", [])

                for idx, agent in enumerate(sub_agents):
                    if isinstance(agent, dict):
                        agent_name = agent.get("name", "SubAgent")
                        agent_model = agent.get("model_name", "N/A")
                        agent_task = agent.get("task", "")
                        agent_tools = agent.get("tools", [])
                    else:
                        agent_name = getattr(agent, "name", "SubAgent")
                        agent_model = getattr(agent, "model_name", "N/A")
                        agent_task = getattr(agent, "task", "")
                        agent_tools = getattr(agent, "tools", [])

                    tool_names = []
                    for t in agent_tools:
                        if isinstance(t, dict):
                            tool_names.append(t.get("tool_title", "tool"))
                        elif hasattr(t, "tool_title"):
                            tool_names.append(t.tool_title)
                        elif hasattr(t, "name"):
                            tool_names.append(t.name)
                        else:
                            tool_names.append(str(t))

                    tools_str = ", ".join(tool_names) if tool_names else "[dim]None[/dim]"
                    layer_label = f"Layer {layer_num}" if idx == 0 else ""
                    plan_table.add_row(layer_label, agent_name, agent_model, agent_task, tools_str)

            console.print(plan_table)
            console.print()

        # 3. Always show the full raw JSON result syntax-highlighted
        try:
            json_str = json.dumps(result, indent=2, ensure_ascii=False)
            syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
            console.print(
                Panel(
                    syntax,
                    title=f"[bold green]🎯 Task Orchestrator Raw Result (JSON): \"{task[:60]}\"[/bold green]",
                    subtitle="[dim]Raw Structured Output[/dim]",
                    border_style="green",
                    box=box.ROUNDED,
                    padding=(1, 2),
                )
            )
        except Exception:
            console.print(
                Panel(
                    str(result),
                    title="[bold green]🎯 Task Orchestrator Raw Result[/bold green]",
                    border_style="green",
                    box=box.ROUNDED
                )
            )
        console.print()

    # =========================================================================
    # 2. ARCHITECTURE TOPOLOGY LOGS
    # =========================================================================

    @classmethod
    def log_executable_layers(cls, executable_layers_list: List[Any]) -> None:
        """Visual tree diagram of generated executable layers and sub-agents with raw details."""
        tree = Tree(f"[bold cyan]🚀 Executable Multi-Agent Architecture ({len(executable_layers_list)} Layers)[/bold cyan]")
        for layer in executable_layers_list:
            layer_num = getattr(layer, "layer_number", "?")
            sub_agents = getattr(layer, "sub_agents", [])
            layer_branch = tree.add(f"[bold yellow]📦 Layer {layer_num}[/bold yellow] [dim]({len(sub_agents)} Parallel Sub-Agents)[/dim]")
            for idx, agent in enumerate(sub_agents, 1):
                agent_tools = getattr(agent, "tools", []) or []
                tool_names = [t.name for t in agent_tools if hasattr(t, "name")]
                tools_str = ", ".join(tool_names) if tool_names else "[dim]None[/dim]"
                is_ready = getattr(agent, "executor", None) is not None
                status = "[green]Ready (Compiled Agent) ✅[/green]" if is_ready else "[red]Failed ❌[/red]"
                sys_prompt = getattr(agent, "system_prompt", "")

                agent_node = layer_branch.add(
                    f"[bold green]Agent #{idx}: {getattr(agent, 'name', 'SubAgent')}[/bold green] "
                    f"[dim]({getattr(agent, 'model_name', 'N/A')})[/dim] - {status}"
                )
                agent_node.add(f"[bold white]Task:[/bold white] {getattr(agent, 'task', 'N/A')}")
                if sys_prompt:
                    agent_node.add(f"[bold cyan]System Prompt:[/bold cyan] [dim]{sys_prompt.strip()}[/dim]")
                agent_node.add(f"[bold blue]Available Tools ({len(tool_names)}):[/bold blue] {tools_str}")

        console.print(
            Panel(
                tree,
                title="[bold bright_blue]ARCHITECTURE & AGENT TOPOLOGY[/bold bright_blue]",
                border_style="bright_blue",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )
        console.print()

    # =========================================================================
    # 3. LAYER LIFECYCLE LOGS
    # =========================================================================

    @classmethod
    def log_layer_start(cls, layer_num: int, sub_agent_count: int) -> None:
        """Visual divider rule for layer execution start."""
        console.print()
        console.rule(
            f"[bold yellow]▶️  [EXECUTING LAYER {layer_num}] - {sub_agent_count} Sub-Agent(s) Running in Parallel[/bold yellow]",
            style="yellow",
        )
        console.print()

    @classmethod
    def log_layer_complete(cls, layer_num: int, result_count: int) -> None:
        """Visual badge for layer completion."""
        console.print(
            f"📦 [bold green][LAYER {layer_num} COMPLETE][/bold green] - "
            f"[dim]{result_count} Sub-Agent Result(s) Recorded[/dim]\n"
        )

    @classmethod
    def log_all_layers_complete(cls, total_layers: int) -> None:
        """Visual divider rule when all layers are completed."""
        console.print()
        console.rule(
            f"[bold bright_green]🏁 ALL {total_layers} LAYER(S) EXECUTED SUCCESSFULLY[/bold bright_green]",
            style="bright_green",
        )
        console.print()

    # =========================================================================
    # 4. SUB-AGENT EXECUTION LOGS
    # =========================================================================

    @classmethod
    def log_subagent_start(
        cls,
        agent_name: str,
        agent_model: str,
        agent_task: str,
        layer_number: Optional[int] = None,
        accumulated_context: Optional[str] = None
    ) -> None:
        """Visual box for starting a sub-agent with layer number and layered feed context."""
        layer_badge = f" [Layer {layer_number}]" if layer_number is not None else ""
        content_lines = [
            f"[bold white]Task:[/bold white] {agent_task}",
            f"[bold white]Model:[/bold white] [dim]{agent_model}[/dim]"
        ]

        # Render layered context feed details if previous layer outputs were injected
        if accumulated_context and accumulated_context.strip():
            content_lines.append("")
            content_lines.append("[bold yellow]📥 Layered Feed (Injected from Prior Layers):[/bold yellow]")
            for line in accumulated_context.strip().splitlines():
                stripped = line.strip()
                if stripped.startswith("[Layer ") and stripped.endswith("]"):
                    content_lines.append(f"  [bold cyan]📂 {stripped}[/bold cyan]")
                elif stripped.startswith("Agent:"):
                    agent_val = stripped[len("Agent:"):].strip()
                    content_lines.append(f"    [bold green]🤖 Agent:[/bold green] [white]{agent_val}[/white]")
                elif stripped.startswith("Task:"):
                    task_val = stripped[len("Task:"):].strip()
                    content_lines.append(f"       [dim yellow]Task:[/dim yellow] [dim]{task_val}[/dim]")
                elif stripped.startswith("Output:"):
                    out_val = stripped[len("Output:"):].strip()
                    preview = (out_val[:120] + "...") if len(out_val) > 120 else out_val
                    content_lines.append(f"       [bold bright_white]Output:[/bold bright_white] [dim cyan]{preview}[/dim cyan]")
                elif stripped:
                    content_lines.append(f"       [dim]{stripped}[/dim]")
        elif layer_number == 1:
            content_lines.append("")
            content_lines.append("[dim italic]ℹ️ Initial Layer 1 (cold start - no prior layer outputs fed)[/dim italic]")

        console.print(
            Panel(
                "\n".join(content_lines),
                title=f"[bold cyan]🚀 [STARTING SUB-AGENT{layer_badge}]: [underline]{agent_name}[/underline][/bold cyan]",
                border_style="cyan",
                box=box.ROUNDED,
                padding=(0, 2),
            )
        )

    @classmethod
    def log_subagent_layered_feed(
        cls,
        agent_name: str,
        layer_number: Optional[int],
        accumulated_context: str
    ) -> None:
        """Visual card dedicated to inspecting the exact previous layer history fed into a sub-agent."""
        if not accumulated_context or not accumulated_context.strip():
            return

        tree = Tree(f"📥 [bold yellow]Layered Context Feed for [cyan]{agent_name}[/cyan] (Layer {layer_number or '?'})[/bold yellow]")
        current_layer_branch = None

        for line in accumulated_context.strip().splitlines():
            line_str = line.strip()
            if line_str.startswith("[Layer ") and line_str.endswith("]"):
                current_layer_branch = tree.add(f"[bold cyan]{line_str}[/bold cyan]")
            elif current_layer_branch is not None:
                if line_str.startswith("Agent:"):
                    current_layer_branch.add(f"[bold green]{line_str}[/bold green]")
                elif line_str.startswith("Output:"):
                    out_val = line_str[len("Output:"):].strip()
                    preview = (out_val[:140] + "...") if len(out_val) > 140 else out_val
                    current_layer_branch.add(f"[dim cyan]Output: {preview}[/dim cyan]")
                elif line_str.startswith("Task:"):
                    current_layer_branch.add(f"[dim]{line_str}[/dim]")
            else:
                tree.add(f"[dim]{line_str}[/dim]")

        console.print(
            Panel(
                tree,
                title="[bold yellow]🧬 Prior Layer Context Injected[/bold yellow]",
                border_style="yellow",
                box=box.ROUNDED,
                padding=(0, 2),
            )
        )

    @classmethod
    def log_subagent_completed(
        cls,
        agent_name: str,
        messages: List[Any],
        final_content: Any
    ) -> None:
        """Visual box for completed sub-agent output and tools used."""
        tool_calls_made = []
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", str(tc))
                    tool_args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                    tool_calls_made.append((tool_name, tool_args))

        if tool_calls_made:
            table = Table(
                title=f"🔧 Tools Invoked by {agent_name} ({len(tool_calls_made)})",
                border_style="magenta",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold magenta"
            )
            table.add_column("Tool Name", style="bold magenta", width=25)
            table.add_column("Arguments (Raw)", style="dim")
            for t_name, t_args in tool_calls_made:
                arg_repr = json.dumps(t_args, ensure_ascii=False) if isinstance(t_args, dict) else str(t_args)
                table.add_row(t_name, arg_repr)
            console.print(table)

        content_str = str(final_content).strip()
        console.print(
            Panel(
                content_str if content_str else "[dim]No content generated[/dim]",
                title=f"[bold green]✅ [COMPLETED SUB-AGENT]: {agent_name}[/bold green]",
                subtitle="[dim]📝 Output Summary (Raw Content)[/dim]",
                border_style="green",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )
        console.print()

    @classmethod
    def log_subagent_failed(cls, agent_name: str, error: Any) -> None:
        """Visual box for sub-agent execution error."""
        console.print(
            Panel(
                f"[bold red]Error: {error}[/bold red]",
                title=f"[bold red]❌ [FAILED SUB-AGENT]: {agent_name}[/bold red]",
                border_style="red",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )

    # =========================================================================
    # 5. DYNAMIC API TOOL EXECUTION LOGS (TOOL GENERATOR)
    # =========================================================================

    @classmethod
    def log_tool_executing(
        cls,
        tool_name: str,
        http_method: str,
        url: str,
        params: Any = None,
        body: Any = None,
        headers: Any = None
    ) -> None:
        """Visual card when a dynamic API tool is dispatched."""
        method_color = "bold green" if http_method.upper() == "GET" else "bold blue"
        req_lines = [
            f"[{method_color}]{http_method.upper()}[/{method_color}] [underline cyan]{url}[/underline cyan]"
        ]
        if params:
            req_lines.append(f"[bold yellow]Query Params (Raw):[/bold yellow] [dim]{json.dumps(params, ensure_ascii=False)}[/dim]")
        if body:
            req_lines.append(f"[bold magenta]Body/Payload (Raw):[/bold magenta] [dim]{json.dumps(body, ensure_ascii=False)}[/dim]")
        if headers:
            req_lines.append(f"[bold cyan]Headers (Raw):[/bold cyan] [dim]{json.dumps(headers, ensure_ascii=False)}[/dim]")

        console.print(
            Panel(
                "\n".join(req_lines),
                title=f"[bold bright_magenta]🔧 [EXECUTE] Dynamic Tool Request: {tool_name}[/bold bright_magenta]",
                border_style="bright_magenta",
                box=box.ROUNDED,
                padding=(0, 2),
            )
        )

    @classmethod
    def log_tool_executed(
        cls,
        tool_name: str,
        http_method: str,
        url: str,
        status_code: Optional[int],
        elapsed_ms: float,
        response_preview: str,
        is_error: bool = False
    ) -> None:
        """Visual card when an API tool completes execution."""
        status_style = "bold green" if (status_code and status_code < 400 and not is_error) else "bold red"
        status_label = f"[{status_style}]HTTP {status_code or 'ERR'}[/{status_style}]"
        latency_label = f"[dim cyan]{elapsed_ms:.1f}ms[/dim cyan]"

        preview = response_preview.strip()
        resp_content = (
            f"[bold white]Status:[/bold white] {status_label} | [bold white]Latency:[/bold white] {latency_label}\n\n"
            f"[bold white]Raw Response:[/bold white]\n[dim]{preview}[/dim]"
        )
        border_style = "green" if not is_error else "red"

        console.print(
            Panel(
                resp_content,
                title=f"[bold {border_style}]⚡ Tool Result: {tool_name}[/bold {border_style}]",
                border_style=border_style,
                box=box.ROUNDED,
                padding=(0, 2),
            )
        )

    # =========================================================================
    # 6. TOOL DISCOVERY WORKFLOW LOGS
    # =========================================================================

    @classmethod
    def log_tool_discovery_start(cls, task: str) -> None:
        """Visual banner for starting the tool discovery workflow."""
        console.print()
        console.rule(
            "[bold magenta]🔍 === Starting Tool Discovery Workflow ===[/bold magenta]",
            style="magenta",
        )
        console.print(
            Panel(
                f"[bold white]Target Task:[/bold white] {task}",
                title="🔍 Tool Discovery Goal",
                border_style="magenta",
                box=box.ROUNDED,
                padding=(0, 2),
            )
        )
        console.print()

    @classmethod
    def log_tool_discovery_analysis(cls, task: str, queries: List[str]) -> None:
        """Visual card for task analysis and extracted search queries."""
        query_items = "\n".join([f"  [bold cyan]•[/bold cyan] [white]{q}[/white]" for q in queries]) if queries else "  [dim]None[/dim]"
        body = (
            f"[bold yellow]Analyzed Task:[/bold yellow] [white]{task}[/white]\n\n"
            f"[bold green]Generated Generic Queries ({len(queries)}):[/bold green]\n"
            f"{query_items}"
        )
        console.print(
            Panel(
                body,
                title="[bold magenta]🔍 [Node: analyze_task] Task Breakdown & Capabilities[/bold magenta]",
                border_style="magenta",
                box=box.ROUNDED,
                padding=(0, 2),
            )
        )

    @classmethod
    def log_tool_discovery_db_search(
        cls,
        queries: List[str],
        tools_found: List[Dict[str, Any]],
        query_details: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Visual table or card for vector database search results."""
        details_text = ""
        if query_details:
            lines = []
            for qd in query_details:
                q = qd.get("query", "")
                v_dim = qd.get("vector_dim", 0)
                m_count = qd.get("matches", 0)
                lines.append(f"  • Query: [cyan]'{q}'[/cyan] | Vector Dim: [green]{v_dim}[/green] | Matches in DB: [yellow]{m_count}[/yellow]")
            details_text = "\n".join(lines)

        if tools_found:
            if details_text:
                console.print(
                    Panel(
                        details_text,
                        title="🗄️ Vector Database Query Breakdown",
                        border_style="cyan",
                        box=box.ROUNDED,
                        padding=(0, 2)
                    )
                )

            table = Table(
                title=f"🗄️ [Node: explore_db] Local Database Matches ({len(tools_found)} Found)",
                border_style="cyan",
                box=box.ROUNDED,
                header_style="bold cyan",
                show_header=True
            )
            table.add_column("Tool Title", style="bold cyan", width=22)
            table.add_column("Method", justify="center", width=8)
            table.add_column("Endpoint URL", style="white")
            table.add_column("Description", style="dim")

            for t in tools_found:
                title = t.get("tool_title", "tool")
                method = t.get("http_method", "GET")
                url = t.get("api_url", "")
                desc = t.get("description", "")
                table.add_row(title, method, url, desc)

            console.print(table)
        else:
            q_info = f"\n\n[bold white]Queries searched ({len(queries)}):[/bold white]\n{details_text}" if details_text else f"\n\n[bold white]Queries searched:[/bold white] {queries}"
            console.print(
                Panel(
                    f"[yellow]No matching existing tools found in local vector DB.[/yellow]{q_info}",
                    title="🗄️ [Node: explore_db] Local Database Search",
                    border_style="dim yellow",
                    box=box.ROUNDED,
                    padding=(0, 2),
                )
            )

    @classmethod
    def log_tool_discovery_web_search(
        cls,
        capability: str,
        query: str,
        results_count: int,
        preview_snippet: str
    ) -> None:
        """Visual card for web search query and raw result snippets."""
        body = (
            f"[bold magenta]Missing Capability:[/bold magenta] [white]{capability}[/white]\n"
            f"[bold cyan]Web Query:[/bold cyan] [underline]{query}[/underline]\n"
            f"[bold green]DuckDuckGo Results Returned:[/bold green] [bold white]{results_count}[/bold white]\n\n"
            f"[bold yellow]--- Search Results Preview (Raw) ---[/bold yellow]\n"
            f"[dim]{preview_snippet.strip()}[/dim]"
        )
        console.print(
            Panel(
                body,
                title="🌐 [Node: web_search] DuckDuckGo API Search",
                border_style="cyan",
                box=box.ROUNDED,
                padding=(0, 2),
            )
        )

    @classmethod
    def log_tool_discovery_evaluation(cls, is_sufficient: bool, missing_capabilities: List[str]) -> None:
        """Visual card for tool sufficiency evaluation."""
        if is_sufficient:
            status = "[bold green]SUFFICIENT ✅[/bold green] [dim]Existing tools satisfy all requirements.[/dim]"
        else:
            status = "[bold yellow]INSUFFICIENT ⚠️[/bold yellow] [dim]External tools must be discovered.[/dim]"

        missing_list = ""
        if missing_capabilities:
            items = "\n".join([f"  [bold red]•[/bold red] [white]{c}[/white]" for c in missing_capabilities])
            missing_list = f"\n\n[bold yellow]Missing Capabilities Needed ({len(missing_capabilities)}):[/bold yellow]\n{items}"

        console.print(
            Panel(
                f"[bold white]Status:[/bold white] {status}{missing_list}",
                title="⚖️ [Node: evaluate_tools] Sufficiency Evaluation",
                border_style="green" if is_sufficient else "yellow",
                box=box.ROUNDED,
                padding=(0, 2),
            )
        )

    @classmethod
    def log_tool_discovery_web_extracted(cls, capability: str, tool_def: Dict[str, Any]) -> None:
        """Visual card for web-searched and LLM-extracted public API definition."""
        tool_title = tool_def.get("tool_title", "API Tool")
        method = tool_def.get("http_method", "GET").upper()
        url = tool_def.get("api_url", "")
        desc = tool_def.get("description", "")
        args = tool_def.get("argument", {})
        headers = tool_def.get("headers", {})
        test_args = tool_def.get("test_arguments", {})

        body = (
            f"[bold magenta]Target Capability:[/bold magenta] [white]{capability}[/white]\n"
            f"[bold cyan]Tool Name:[/bold cyan] [bold white]{tool_title}[/bold white]\n"
            f"[bold green]Endpoint:[/bold green] [{ 'bold green' if method == 'GET' else 'bold blue' }]{method}[/] [underline]{url}[/underline]\n"
            f"[bold white]Description:[/bold white] [dim]{desc}[/dim]\n"
            f"[bold yellow]Arguments Schema (Raw):[/bold yellow] [dim]{json.dumps(args, ensure_ascii=False)}[/dim]\n"
            f"[bold cyan]Headers (Raw):[/bold cyan] [dim]{json.dumps(headers, ensure_ascii=False)}[/dim]\n"
            f"[bold blue]Sample Test Args:[/bold blue] [dim]{json.dumps(test_args, ensure_ascii=False)}[/dim]"
        )
        console.print(
            Panel(
                body,
                title=f"🌐 [Node: web_search] Extracted Free Public API: [underline]{tool_title}[/underline]",
                border_style="magenta",
                box=box.ROUNDED,
                padding=(0, 2),
            )
        )

    @classmethod
    def log_tool_discovery_test(
        cls,
        tool_title: str,
        test_args: Dict[str, Any],
        passed: bool,
        result_preview: str
    ) -> None:
        """Visual card for testing the discovered web API tool."""
        status_tag = "[bold green]PASSED ✅[/bold green]" if passed else "[bold red]FAILED ❌[/bold red]"
        border = "green" if passed else "red"

        body = (
            f"[bold white]Test Status:[/bold white] {status_tag}\n"
            f"[bold cyan]Test Arguments (Raw):[/bold cyan] [dim]{json.dumps(test_args, ensure_ascii=False)}[/dim]\n"
            f"[bold white]Execution Output (Raw):[/bold white]\n[dim]{result_preview.strip()}[/dim]"
        )
        console.print(
            Panel(
                body,
                title=f"🧪 [Node: test_tool] Validation: [underline]{tool_title}[/underline]",
                border_style=border,
                box=box.ROUNDED,
                padding=(0, 2),
            )
        )

    @classmethod
    def log_tool_discovery_stored(cls, stored_count: int) -> None:
        """Visual notification for caching validated tools to DB."""
        if stored_count > 0:
            console.print(
                Panel(
                    f"[bold green]Successfully saved {stored_count} tools to DB.[/bold green]",
                    title="💾 [Node: store_tool] Database Caching",
                    border_style="green",
                    box=box.ROUNDED,
                    padding=(0, 2),
                )
            )
        else:
            console.print(
                Panel(
                    "[dim]No new tools saved to DB.[/dim]",
                    title="💾 [Node: store_tool] Database Caching",
                    border_style="dim",
                    box=box.ROUNDED,
                    padding=(0, 2),
                )
            )

    @classmethod
    def log_tool_discovery_final(cls, final_answer: str) -> None:
        """Visual panel for tool discovery final synthesis."""
        console.print(
            Panel(
                str(final_answer).strip(),
                title="🎯 [Node: final_answer] Tool Discovery Synthesis",
                border_style="gold1",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )
        console.print()

    @classmethod
    def log_tool_discovery_complete(cls, tools_found: List[Dict[str, Any]]) -> None:
        """Visual summary table of discovered tools."""
        console.print()
        if tools_found:
            table = Table(
                title=f"✅ Tool Discovery Complete ({len(tools_found)} Tools Ready)",
                border_style="bright_green",
                box=box.ROUNDED,
                header_style="bold bright_green",
                show_header=True
            )
            table.add_column("Tool Title", style="bold cyan", width=24)
            table.add_column("Method", justify="center", width=8)
            table.add_column("Endpoint URL", style="white")
            table.add_column("Description", style="dim")

            for t in tools_found:
                title = t.get("tool_title", "tool")
                method = t.get("http_method", "GET")
                url = t.get("api_url", "")
                desc = t.get("description", "")
                table.add_row(title, method, url[:40] + "..." if len(url) > 40 else url, desc[:50] + "..." if len(desc) > 50 else desc)

            console.print(table)
        else:
            console.print("[yellow]⚠️  Tool Discovery Finished: No external tools needed or found.[/yellow]")
        console.print()

    # =========================================================================
    # 7. FINAL SYNTHESIS & WARNING LOGS
    # =========================================================================

    @classmethod
    def log_final_response(cls, final_response: str) -> None:
        """Visual highlight panel for the final workflow response."""
        console.print(
            Panel(
                str(final_response).strip(),
                title="[bold gold1]🎯 FINAL SYNTHESIZED WORKFLOW RESPONSE[/bold gold1]",
                border_style="gold1",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )
        console.print()

    @classmethod
    def log_warning(cls, message: str) -> None:
        """Visual warning alert."""
        console.print(f"[bold yellow]⚠️  WARNING:[/bold yellow] {message}")
