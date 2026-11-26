"""Turn explicit tool failure messages into exceptions for the execution handoff."""
import json
from langchain.agents.middleware import AgentMiddleware


class ToolExecutionError(RuntimeError):
    def __init__(self, tool_name, content, messages):
        self.tool_name = tool_name
        self.messages = messages
        super().__init__(f"Tool {tool_name!r} failed: {content}")


def raise_tool_message_error(messages):
    """Inspect tool messages only; ordinary assistant prose is not a failure signal."""
    for message in messages:
        def field(name, default=None):
            return message.get(name, default) if isinstance(message, dict) else getattr(message, name, default)

        if field("type") not in ("tool", "ToolMessage") and field("role") != "tool":
            continue
        content = field("content", "")
        artifact = field("artifact")
        failed = field("status") == "error"
        if isinstance(artifact, dict):
            failed = failed or artifact.get("isError") is True or artifact.get("is_error") is True
        for block in content if isinstance(content, list) else [content]:
            value = block
            if isinstance(value, dict) and value.get("type") == "text":
                value = value.get("text", "")
            if isinstance(value, str):
                failed = failed or value.lstrip().startswith("Error executing tool")
                try:
                    value = json.loads(value)
                except (ValueError, TypeError):
                    continue
            if isinstance(value, dict):
                failed = failed or value.get("isError") is True or value.get("is_error") is True
                failed = failed or value.get("status") == "error" or bool(value.get("error"))
        if failed:
            raise ToolExecutionError(field("name") or "unknown", content, messages)


class RaiseToolErrorsMiddleware(AgentMiddleware):
    """Stop after a tool failure before the next model call."""
    def before_model(self, state, runtime):
        raise_tool_message_error(state.get("messages", []))

    async def abefore_model(self, state, runtime):
        raise_tool_message_error(state.get("messages", []))
