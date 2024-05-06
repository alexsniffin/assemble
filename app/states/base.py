import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List

import tenacity

from app.llm.generator import Generator
from app.llm.types import Usage
from app.memory.memory import Memory
from app.memory.scratch_pad import ContextException
from app.persona.persona import Persona
from app.states.states import SystemStates
from app.tools.adapter import ToolAdapter

logger = logging.getLogger(__name__)


@dataclass
class Transition:
    next_state: str


@dataclass
class StateResult:
    transition: Transition
    prompt: str
    response: str
    token_usage: Usage


class StateBase(ABC):
    """ Abstract base class for all nodes in the state machine. """

    name: str

    def __init__(self,
                 generator: Generator,
                 tools: Optional[List[ToolAdapter]] = None,
                 retry_attempts: int = 5,
                 retry_multiplier: int = 2,
                 retry_min: int = 2,
                 retry_max: int = 30):
        self.generator = generator
        self.tools = tools
        self.retry = tenacity.retry(
            stop=tenacity.stop_after_attempt(retry_attempts),
            wait=tenacity.wait_exponential(multiplier=retry_multiplier, min=retry_min, max=retry_max),
            retry=tenacity.retry_if_exception_type(Exception),
            reraise=True,
            retry_error_callback=lambda retry_state: logger.error(
                f"Retry failed after {retry_state.attempt_number} attempts: {retry_state.outcome.exception()}"),
            after=tenacity.after_log(logger, logging.INFO),
            before=tenacity.before_log(logger, logging.INFO)
        )
        self._context_handler_limit = 50

    @abstractmethod
    def build_prompt(self,
                     persona: Persona,
                     memory: Memory,
                     tools: Optional[List[ToolAdapter]]) -> str:
        pass

    @abstractmethod
    def handle_transition(self,
                          response: str,
                          memory: Memory,
                          tools: Optional[List[ToolAdapter]]) -> Transition:
        pass

    def execute(self, persona: Persona, memory: Memory) -> StateResult:
        """ Execute the state logic, including generating responses and handling state transitions. """

        @self.retry
        def _execute_with_retry():
            prompt = None
            try:
                for _ in range(self._context_handler_limit):
                    prompt = self.build_prompt(persona, memory, self.tools)
                    if self.generator.is_context_limit(prompt):
                        memory.run_context_handlers()
                    else:
                        break
            except ContextException as e:
                logger.error(f"Failed to generate LLM response: {e}")
                return prompt, {}, Transition(
                    next_state=SystemStates.EXIT.value
                )

            response, token_usage = self.generator.generate(prompt)
            transition = self.handle_transition(response, memory, self.tools)
            return StateResult(transition, prompt, response, token_usage)

        return _execute_with_retry()
