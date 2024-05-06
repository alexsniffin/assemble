from abc import ABC
from typing import Optional, List

from app.llm.generator import Generator
from app.memory.memory import Memory
from app.persona.persona import Persona
from app.states.base import StateBase
from app.states.defaults import DefaultTextHandlerBase
from app.states.states import SystemStates
from app.tools.adapter import ToolAdapter


class FinalAnswerState(DefaultTextHandlerBase, StateBase):
    name: str = "final_answer"

    def __init__(self, generator: Generator, tools: Optional[List[ToolAdapter]] = None):
        StateBase.__init__(self, generator, tools)
        DefaultTextHandlerBase.__init__(self, next_state=SystemStates.EXIT.value, exclude_from_scratch_pad=False)

    def build_prompt(self,
                     persona: Persona,
                     memory: Memory,
                     tools: Optional[List[ToolAdapter]]) -> str:
        return f'''{persona.prompt()}

        Given the problem from the user, use your notes to give an answer. Directly address the problems.

        Problem:
        """
        {memory.data.get_current_message().content}
        """

        Notes:
        """
        {memory.scratch_pad.prompt()}
        """

        Your answer to the problem.'''
