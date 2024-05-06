from typing import Optional, List

from app.core.memory.memory import Memory
from app.core.states.base import Transition
from app.core.tools.adapter import ToolAdapter, ToolBase


def text_handler(response: str,
                 memory: Memory,
                 next_state: str,
                 save_data_key: Optional[str] = None, ) -> Transition:
    if save_data_key is not None:
        memory.data.set(save_data_key, response)
    return Transition(
        next_state=next_state
    )


def tools_handler(
        response: str,
        memory: Memory,
        tools: Optional[List[ToolAdapter]],
        next_state: str,
        save_data_key: Optional[str] = None) -> Transition:
    if tools is None:
        raise ValueError("Tools must be provided for this state handler.")

    parsed_tool_input = ToolBase.model_validate_json(response)
    tool = next((tool for tool in tools if tool.name == parsed_tool_input.tool_name), None)
    if tool is None:
        return Transition(
            updated_response="Invalid tool name for response. Please try again.",
            next_state=next_state,
        )

    tool_input = tool.validate(response)
    output = tool.run(tool_input)

    if tool.exclude_from_scratch_pad:
        tool_results = f'Tool executed for {tool.name}.'
    else:
        tool_results = f'Tool results for {tool.name}: """\nInput: {response}\nOutput: {output.model_dump_json()}"""'
    if save_data_key is not None:
        memory.data.set(save_data_key, tool_results)
    return Transition(
        updated_response=tool_results,
        next_state=next_state,
    )
