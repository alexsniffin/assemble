from typing import List, Optional

from app.core.llm.generator import Generator
from app.core.memory.memory import Memory
from app.core.persona import Persona
from app.core.states.base import StateBase, Transition
from app.core.tools.adapter import ToolAdapter
from app.states.defaults import text_handler


class SummarizeMessagesState(StateBase):
    name: str = "summarize_messages"

    def __init__(self,
                 next_state: str,
                 generator: Generator,
                 tools: Optional[List[ToolAdapter]] = None,
                 retry_attempts: int = 5,
                 retry_multiplier: int = 2,
                 retry_min: int = 2,
                 retry_max: int = 30):
        super().__init__(
            generator=generator,
            tools=tools,
            retry_attempts=retry_attempts,
            retry_multiplier=retry_multiplier,
            retry_min=retry_min,
            retry_max=retry_max
        )
        self.next_state = next_state

    def build_prompt(self,
                     persona: Persona,
                     memory: Memory,
                     tools: Optional[List[ToolAdapter]]) -> str:
        messages_formatted = "\n".join(
            [f"{message.name}: {message.content}\n" for message in memory.data.get_all_messages()])
        return f'''Given the conversation history, summarize the messages.

Messages:
"""
{messages_formatted}
"""

Summarize:'''

    def before_generation(self,
                          memory: Memory,
                          tools: Optional[List[ToolAdapter]]) -> Optional[Transition]:
        if len(memory.data.get_all_messages()) == 1:
            return Transition(next_state=self.next_state)

    def after_generation(self, response: str, memory: Memory, tools: Optional[List[ToolAdapter]]) -> Transition:
        return text_handler(response=response, memory=memory, next_state=self.next_state)
