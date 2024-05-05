import logging
from dataclasses import dataclass
from typing import Callable, Optional, List, Dict, Any, Tuple
import tenacity

from app.llm.generator import Generator
from app.memory.memory import Memory
from app.memory.scratch_pad import ContextException
from app.persona.persona import Persona
from app.state.states import SystemStates
from app.tools.adapter import ToolAdapter

logger = logging.getLogger(__name__)


@dataclass
class GenerationHandlerResponse:
    output: str
    exclude_from_scratch_pad: bool
    next_state: str


class Node:
    """ Represents a state in the agent's state machine, handling state-specific logic and transitions. """

    def __init__(self,
                 state_name: str,
                 prompt_handler: Callable[[Persona, Memory, Optional[List[ToolAdapter]]], str],
                 generation_handler: Callable[[str, Memory, Optional[List[ToolAdapter]]], GenerationHandlerResponse],
                 generator: Generator,
                 tools: Optional[List[ToolAdapter]] = None,
                 retry_attempts: int = 5,
                 retry_multiplier: int = 2,
                 retry_min: int = 2,
                 retry_max: int = 30):
        self.state_name = state_name
        self.prompt_handler = prompt_handler
        self.generation_handler = generation_handler
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

    def execute(self, persona: Persona, memory: Memory) -> Tuple[str, Dict[str, int], GenerationHandlerResponse]:
        """ Execute the state logic, including generating responses and handling state transitions. """

        @self.retry
        def _execute_with_retry():
            prompt = self.prompt_handler(persona, memory, self.tools)
            try:
                for _ in range(self._context_handler_limit):
                    if self.generator.is_context_limit(prompt):
                        memory.run_context_handlers()
                        prompt = self.prompt_handler(persona, memory, self.tools)
                    else:
                        break
            except ContextException as e:
                logger.error(f"Failed to generate LLM response: {e}")
                return prompt, {}, GenerationHandlerResponse(
                    output="Unable to generate anymore.",
                    exclude_from_scratch_pad=True,
                    next_state=SystemStates.EXIT.value
                )

            response, token_usage = self.generator.generate(prompt)
            state_handler_response = self.generation_handler(response, memory, self.tools)
            return prompt, token_usage, state_handler_response

        return _execute_with_retry()
