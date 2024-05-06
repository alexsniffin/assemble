from abc import ABC
from typing import Optional, List

from app.memory.memory import Memory
from app.states.base import Transition
from app.tools.adapter import ToolAdapter, ToolBase


class DefaultTextHandlerBase(ABC):
    def __init__(self, next_state: str, exclude_from_scratch_pad: bool, save_data_key: Optional[str] = None):
        self.next_state = next_state
        self.exclude_from_scratch_pad = exclude_from_scratch_pad
        self.save_data_key = save_data_key

    def handle_transition(self,
                          response: str,
                          memory: Memory,
                          _: Optional[List[ToolAdapter]]) -> Transition:
        if self.save_data_key is not None:
            memory.data.set(self.save_data_key, response)
        return Transition(next_state=self.next_state)


class DefaultToolsHandlerBase(ABC):
    def __init__(self, next_state: str, exclude_from_scratch_pad: bool, save_data_key: Optional[str] = None):
        self.next_state = next_state
        self.exclude_from_scratch_pad = exclude_from_scratch_pad
        self.save_data_key = save_data_key

    def handle_transition(self,
                          response: str,
                          memory: Memory,
                          tools: Optional[List[ToolAdapter]]) -> Transition:
        if tools is None:
            raise ValueError("Tools must be provided for this state handler.")

        parsed_tool_input = ToolBase.model_validate_json(response)
        tool = next((tool for tool in tools if tool.name == parsed_tool_input.tool_name), None)
        if tool is None:
            return Transition(next_state=self.next_state)

        tool_input = tool.validate(response)
        output = tool.run(tool_input)

        tool_results = f"Tool input: {response}\nTool output: {output.model_dump_json()}"
        if self.save_data_key is not None:
            memory.data.set(self.save_data_key, tool_results)
        return Transition(next_state=self.next_state)
