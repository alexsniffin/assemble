from typing import Optional, Callable, List

from app.memory.memory import Memory
from app.state.node import GenerationHandlerResponse
from app.tools.adapter import ToolAdapter, ToolBase


def default_text_handler(
        next_state: str,
        exclude_from_scratch_pad: bool,
        save_data_key: Optional[str] = None) -> Callable[
    [str, Memory, Optional[List[ToolAdapter]]], GenerationHandlerResponse]:
    def handler(response: str, memory: Memory, _: Optional[List[ToolAdapter]]) -> GenerationHandlerResponse:
        if save_data_key is not None:
            memory.data.set(save_data_key, response)
        return GenerationHandlerResponse(
            output=response,
            exclude_from_scratch_pad=exclude_from_scratch_pad,
            next_state=next_state
        )

    return handler


def default_tools_handler(
        next_state: str,
        exclude_from_scratch_pad: bool,
        save_data_key: Optional[str] = None) -> Callable[[str, Memory, List[ToolAdapter]], GenerationHandlerResponse]:
    def handler(response: str, memory: Memory, tools: Optional[List[ToolAdapter]]) -> GenerationHandlerResponse:
        if tools is None:
            raise ValueError("Tools must be provided for this state handler.")

        parsed_tool_input = ToolBase.model_validate_json(response)
        tool = next((tool for tool in tools if tool.name == parsed_tool_input.tool_name), None)
        if tool is None:
            return GenerationHandlerResponse(
                output="Invalid tool name for response. Please try again.",
                next_state=next_state,
                exclude_from_scratch_pad=exclude_from_scratch_pad,
            )

        tool_input = tool.validate(response)
        output = tool.run(tool_input)

        tool_results = f"Tool input: {response}\nTool output: {output.model_dump_json()}"
        if save_data_key is not None:
            memory.data.set(save_data_key, tool_results)
        return GenerationHandlerResponse(
            output=tool_results,
            next_state=next_state,
            exclude_from_scratch_pad=exclude_from_scratch_pad,
        )

    return handler
