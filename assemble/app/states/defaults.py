import json
from typing import Optional, List

from pydantic import BaseModel
from pydantic.v1 import BaseModel as BaseModelV1

from assemble.app.core.memory.memory import Memory
from assemble.app.core.states.base import Transition
from assemble.app.core.tools.adapter import ToolAdapter


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

    parsed_tool_input = json.loads(response)
    tool = next((tool for tool in tools if tool.name == parsed_tool_input['tool_name']), None)
    if tool is None:
        return Transition(
            updated_response="Invalid tool name for response. Please try again.",
            next_state=next_state,
        )

    del parsed_tool_input['tool_name']

    typed_input = tool.validate(parsed_tool_input)
    output = tool.run(typed_input)

    tool_results = f'Tool executed for {tool.name}.'
    if not tool.exclude_input_from_scratch_pad:
        tool_results = tool_results + f'\nInput: {response}'
    if not tool.exclude_output_from_scratch_pad:
        if isinstance(output, str):
            tool_results = tool_results + f'\nOutput: {output}'
        elif isinstance(output, BaseModel):
            tool_results = tool_results + f'\nOutput: {output.model_dump_json()}'
        elif isinstance(output, BaseModelV1):
            tool_results = tool_results + f'\nOutput: {output.json()}'
        else:
            tool_results = tool_results + f'\nOutput: {str(output)}'
    if save_data_key is not None:
        memory.data.set(save_data_key, tool_results)
    return Transition(
        updated_response=tool_results,
        next_state=next_state,
    )
