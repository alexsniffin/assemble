import logging
import threading
from typing import Dict, List, Any, Optional

import pykka
from pydantic import BaseModel
from pykka import ThreadingFuture

from app.core.messages import Query
from app.core.memory.memory import Memory, Message
from app.core.persona import Persona
from app.core.states.base import StateBase
from app.core.states.states import SystemStates
from app.core.types import Usage

logger = logging.getLogger(__name__)


class Step(BaseModel):
    state_name: str
    next_state: str
    prompt: Optional[str] = None
    output: Optional[str] = None
    token_usage: Optional[Usage] = None


class Response(BaseModel):
    final_output: str
    metadata: Dict[str, Any]


class AgentBase(pykka.ThreadingActor):
    name: str
    description: str

    def __init__(self,
                 persona: Persona,
                 memory: Memory,
                 states: List[StateBase],
                 default_initial_state: str,
                 step_limit_state_name: str,
                 clear_scratch_pad_after_answer: bool = False,
                 clear_data_after_answer: bool = False,
                 step_limit: int = 10):
        super().__init__()
        if len(states) == 0:
            raise ValueError("At least one state must be provided.")

        self.persona: Persona = persona
        self.memory: Memory = memory
        self.states: Dict[str, StateBase] = {
            state.name: state
            for state in states
        }
        self.current_state: str = SystemStates.IDLE.value
        self.default_initial_state: str = default_initial_state
        self.clear_scratch_pad_after_answer: bool = clear_scratch_pad_after_answer
        self.clear_data_after_answer: bool = clear_data_after_answer
        self.step_limit: int = step_limit
        self.step_limit_state_name: str = step_limit_state_name

    def on_receive(self, message):
        logger.info(f"Agent {self.__class__.__name__}:{self.actor_urn} received message: {message}")
        if isinstance(message, Query):
            future = ThreadingFuture()

            def thread_func():
                try:
                    steps = self._run(
                        goal=message.goal,
                        from_caller=message.from_caller,
                        initial_state=message.initial_state)
                    response = Response(final_output=steps[-1].output, metadata={
                        "steps": steps,
                    })
                    future.set(response)
                except Exception as e:
                    future.set_exception(e)
                    logger.error(f"Error during message processing for {self.__class__.__name__}:{self.actor_urn}: {e}")

            threading.Thread(target=thread_func).start()
            return future
        else:
            logger.error(f"Unexpected message for {self.__class__.__name__}:{self.actor_urn}: {message}")
            raise ValueError(f"Unexpected message for {self.__class__.__name__}:{self.actor_urn}: {message}")

    def _run(self,
             goal: str,
             from_caller: str,
             initial_state: str = None) -> List[Step]:
        if initial_state is None:
            initial_state = self.default_initial_state

        step_count = 0
        steps = []
        self.current_state = initial_state

        self.memory.data.add_message(Message(name=from_caller, content=goal))

        while self.current_state in self.states:
            if self.current_state == SystemStates.EXIT.value:
                break
            elif step_count > self.step_limit:
                self.current_state = self.step_limit_state_name

            state = self.states[self.current_state]
            logger.info(f"Agent {self.__class__.__name__}:{self.actor_urn} executing state: {self.current_state}")

            response = state.execute(self.persona, self.memory)
            if response is None:
                raise ValueError(
                    f"Invalid state transition from: {self.current_state}.")

            self.memory.scratch_pad.set(f"{state.name.upper()}: {response.response}")

            next_state = response.next_state
            if self.current_state == self.step_limit_state_name:
                next_state = SystemStates.EXIT.value

            self.current_state = next_state
            if self.current_state != SystemStates.EXIT.value and not self.states[self.current_state]:
                raise ValueError(
                    f"Invalid state transition: {self.current_state} -> {response.next_state}. Missing "
                    f"state node for next state of {self.current_state}.")

            steps.append(
                Step(
                    state_name=f"{state.name}",
                    prompt=response.prompt,
                    output=response.response,
                    next_state=f"{self.current_state}",
                    token_usage=response.token_usage
                )
            )
            step_count += 1

        if self.clear_scratch_pad_after_answer:
            self.memory.reset_scratch_pad()
            logger.info(f"Agent {self.__class__.__name__}:{self.actor_urn} cleared scratch pad.")
        elif self.clear_data_after_answer:
            self.memory.reset_data()
            logger.info(f"Agent {self.__class__.__name__}:{self.actor_urn} cleared data.")

        if len(steps) == 0:
            raise ValueError(f"Agent {self.__class__.__name__}:{self.actor_urn} failed to execute, initial state "
                             f"{initial_state} not found.")

        self.memory.data.add_message(Message(name=self.__class__.__name__, content=steps[-1].output))
        logger.info(f"Agent {self.__class__.__name__}:{self.actor_urn} finished execution.")
        return steps
