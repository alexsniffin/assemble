from typing import Optional, List

from app.llm.generator import Generator
from app.memory.memory import Memory
from app.persona.persona import Persona
from app.state.final_answer.constants import FINAL_ANSWER_STATE
from app.state.node import GenerationHandlerResponse, Node
from app.state.states import SystemStates

from app.tools.adapter import ToolAdapter


def final_answer_generation_handler(response: str, _: Memory,
                                    __: Optional[List[ToolAdapter]]) -> GenerationHandlerResponse:
    return GenerationHandlerResponse(
        output=response,
        exclude_from_scratch_pad=False,
        next_state=SystemStates.EXIT.value,
    )


def final_answer_prompt_handler(persona: Persona, memory: Memory, _: Optional[List[ToolAdapter]]) -> str:
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


class FinalAnswerNodeFactory:

    @staticmethod
    def get_node(generator: Generator):
        return Node(
            state_name=FINAL_ANSWER_STATE,
            prompt_handler=final_answer_prompt_handler,
            generation_handler=final_answer_generation_handler,
            generator=generator
        )
