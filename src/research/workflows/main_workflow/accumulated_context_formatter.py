"""
Accumulated Context Formatter.

Utility functions for formatting execution checkpoints into structured accumulated context
to pass across sequential execution layers in the workflow.
"""

from typing import List, Optional, Any


def format_accumulated_context_from_checkpoints(
    checkpoints: Optional[List[Any]],
    current_layer: Optional[int] = None
) -> str:
    """
    Extracts and formats previous completed layer outputs from checkpoints into accumulated context text.
    
    Args:
        checkpoints: List of CheckpointDetails from execution layers.
        current_layer: The currently running layer number (to only include previous layers).
        
    Returns:
        Formatted context string matching '[Layer X Outputs]\nAgent: ...\nTask: ...\nOutput: ...'
    """
    if not checkpoints:
        return ""

    layer_blocks: List[str] = []
    for cp in checkpoints:
        # Resolve layer number
        cp_layer = cp.get("layer_number") if isinstance(cp, dict) else getattr(cp, "layer_number", None)

        # Only include prior layers
        if current_layer is not None and cp_layer is not None and cp_layer >= current_layer:
            continue

        agents_list = cp.get("agents", []) if isinstance(cp, dict) else getattr(cp, "agents", [])
        if not agents_list:
            continue

        agent_parts: List[str] = []
        for agent_cp in agents_list:
            if isinstance(agent_cp, dict):
                a_name = agent_cp.get("agent", "")
                a_task = agent_cp.get("task", "")
                a_out = agent_cp.get("output", "")
            else:
                a_name = getattr(agent_cp, "agent", "")
                a_task = getattr(agent_cp, "task", "")
                a_out = getattr(agent_cp, "output", "")

            agent_parts.append(
                f"Agent: {a_name}\n"
                f"Task: {a_task}\n"
                f"Output: {a_out}"
            )

        layer_num_str = cp_layer if cp_layer is not None else "?"
        layer_block = f"[Layer {layer_num_str} Outputs]\n" + "\n\n".join(agent_parts)
        layer_blocks.append(layer_block)

    return "\n\n".join(layer_blocks)
