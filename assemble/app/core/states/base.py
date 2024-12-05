import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List

import tenacity

from assemble.app.core.llm.generator import Generator
from assemble.app.core.memory.memory import Memory
from assemble.app.core.memory.scratch_pad import ContextException
from assemble.app.core.persona import Persona
from assemble.app.core.states.states import SystemStates
from assemble.app.core.tools.adapter import ToolAdapter
from assemble.app.core.types import Usage

logger = logging.getLogger(__name__)


@dataclass
class Transition:
    next_state: str
    updated_response: Optional[str] = None
    token_usage: Optional[Usage] = None


@dataclass
class StateResponse:
    next_state: str
    prompt: Optional[str] = None
    response: Optional[str] = None
    token_usage: Optional[Usage] = None


class StateBase(ABC):
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

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, 'name') or not isinstance(cls.name, str) or not cls.name.strip():
            logger.error(f"Class {cls.__name__} is missing a static 'name' attribute, it is not a string, or it is "
                         f"empty.")
            raise ValueError(f"Class {cls.__name__} must have a non-empty static 'name' string attribute.")

    @abstractmethod
    def build_prompt(self,
                     persona: Persona,
                     memory: Memory,
                     tools: Optional[List[ToolAdapter]]) -> str:
        pass

    def before_generation(self,
                          memory: Memory,
                          tools: Optional[List[ToolAdapter]]) -> Optional[Transition]:
        return None

    @abstractmethod
    def after_generation(self,
                         generation: str,
                         memory: Memory,
                         tools: Optional[List[ToolAdapter]]) -> Transition:
        pass

    def execute(self, persona: Persona, memory: Memory) -> StateResponse:
        @self.retry
        def _execute_with_retry():
            transition = self.before_generation(memory, self.tools)
            if transition is not None:
                return StateResponse(transition.next_state)

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

            transition = self.after_generation(response, memory, self.tools)
            if transition.updated_response is not None:
                response = transition.updated_response

            if transition.token_usage is not None:
                token_usage.total_tokens += transition.token_usage.total_tokens
                token_usage.completion_tokens += transition.token_usage.completion_tokens
                token_usage.prompt_tokens += transition.token_usage.prompt_tokens

            return StateResponse(transition.next_state, prompt, response, token_usage)

        return _execute_with_retry()
