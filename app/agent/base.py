import logging
import threading
from typing import Dict, List

import pykka
from pydantic import BaseModel
from pykka import ThreadingFuture

from app.agent.messages import Query
from app.memory.memory import Memory, Message
from app.persona.persona import Persona
from app.state.node import Node
from app.state.states import SystemStates

logger = logging.getLogger(__name__)


class Step(BaseModel):
    name: str
    prompt: str
    output: str
    next: str
    token_usage: Dict[str, int]


class Response(BaseModel):
    output: str
    steps: List[Step]


class BaseAgent(pykka.ThreadingActor):
    name: str
    description: str

    def __init__(self,
                 persona: Persona,
                 memory: Memory,
                 nodes: List[Node],
                 default_initial_state: str,
                 step_limit_state_name: str,
                 clear_scratch_after_answer: bool = False,
                 clear_data_after_answer: bool = False,
                 step_limit: int = 10):
        super().__init__()
        if len(nodes) == 0:
            raise ValueError("At least one state must be provided.")

        self.persona = persona
        self.memory = memory
        self.states = {node.state_name: node for node in nodes}
        self.current_state = SystemStates.IDLE.value
        self.default_initial_state = default_initial_state
        self.clear_scratch_after_answer = clear_scratch_after_answer
        self.clear_data_after_answer = clear_data_after_answer
        self.step_limit = step_limit
        self.step_limit_state_name = step_limit_state_name

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
                    response = Response(output=steps[-1].output, steps=steps)
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

        self.memory.data.add_message(Message(name=from_caller, content=goal))
        step_count = 0
        history = []
        self.current_state = initial_state

        while self.current_state in self.states:
            if self.current_state == SystemStates.EXIT.value:
                break
            elif step_count > self.step_limit:
                self.current_state = self.step_limit_state_name

            state = self.states[self.current_state]
            logger.info(f"Agent {self.__class__.__name__}:{self.actor_urn} executing state: {self.current_state}")

            prompt, token_usage, response = state.execute(self.persona, self.memory)
            if not response.exclude_from_scratch_pad:
                self.memory.scratch_pad.set(f"{state.state_name}: {response.output}")

            next_state = response.next_state
            if self.current_state == self.step_limit_state_name:
                next_state = SystemStates.EXIT.value

            self.current_state = next_state
            if self.current_state != SystemStates.EXIT.value and not self.states[self.current_state]:
                raise ValueError(f"Invalid state transition: {self.current_state} -> {response.next_state}. Missing "
                                 f"state node for next state of {self.current_state}.")

            history.append(Step(name=state.state_name, prompt=prompt, output=response.output, next=self.current_state,
                                token_usage=token_usage))
            step_count += 1

        if self.clear_scratch_after_answer:
            self.memory.reset_scratch_pad()
            logger.info(f"Agent {self.__class__.__name__}:{self.actor_urn} cleared scratch pad.")
        elif self.clear_data_after_answer:
            self.memory.reset_data()
            logger.info(f"Agent {self.__class__.__name__}:{self.actor_urn} cleared data.")

        if len(history) == 0:
            raise ValueError(f"Agent {self.__class__.__name__}:{self.actor_urn} failed to execute, initial state "
                             f"{initial_state} not found.")

        self.memory.data.add_message(Message(name=self.__class__.__name__, content=history[-1].output))
        logger.info(f"Agent {self.__class__.__name__}:{self.actor_urn} finished execution.")
        return history
