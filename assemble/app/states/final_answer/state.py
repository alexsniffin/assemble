from typing import Optional, List

from assemble.app.core.memory.memory import Memory
from assemble.app.core.persona import Persona
from assemble.app.core.states.base import StateBase, Transition
from assemble.app.states.defaults import text_handler
from assemble.app.core.states.states import SystemStates
from assemble.app.core.tools.adapter import ToolAdapter


class FinalAnswerState(StateBase):
    name: str = "final_answer"

    def build_prompt(self,
                     persona: Persona,
                     memory: Memory,
                     tools: Optional[List[ToolAdapter]]) -> str:
        return f'''{persona.prompt()}

Given the problem from the user, use your notes to give an answer. Directly address the problem.

Problem:
"""
{memory.data.get_current_message().content}
"""

Notes from oldest to newest:
"""
{memory.scratch_pad.prompt()}
"""

Your complete and detailed answer to the problem.'''

    def after_generation(self, response: str, memory: Memory, tools: Optional[List[ToolAdapter]]) -> Transition:
        return text_handler(response=response, memory=memory, next_state=SystemStates.EXIT.value)
