from typing import List, Optional

from assemble.app.core.llm.generator import Generator
from assemble.app.core.memory.memory import Memory
from assemble.app.core.persona import Persona
from assemble.app.core.states.base import StateBase, Transition
from assemble.app.core.tools.adapter import ToolAdapter
from assemble.app.states.defaults import text_handler


class RewriteQuestionState(StateBase):
    name: str = "rewrite_question"

    def __init__(self,
                 user_name: str,
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
        self.user_name = user_name
        self.next_state = next_state

    def build_prompt(self,
                     persona: Persona,
                     memory: Memory,
                     tools: Optional[List[ToolAdapter]]) -> str:
        messages_formatted = "\n".join(
            [
                f"{user_message.name}: {user_message.content}\n"
                for user_message in memory.data.get_all_messages()
            ])
        return f'''Rewrite the last message from {self.user_name} into a single, coherent statement using the context from the conversation history.

Please focus solely on rewriting the message clearly; do not respond to any queries it contains. Do not summarize, simply capture the subject being discussed to best rewrite the last message.

Example of a good rewrite:
"""
user: What color is the sky?
assistant: The sky is blue.
user: Why is it that color?
Rewritten message: Why is the sky blue?
"""

Example of a bad rewrite:
"""
user: What color is the sky?
assistant: The sky is blue.
user: Why is it that color?
Rewritten message: The sky is blue because of Rayleigh scattering.
"""

{messages_formatted}
Rewritten message: '''

    def before_generation(self,
                          memory: Memory,
                          tools: Optional[List[ToolAdapter]]) -> Optional[Transition]:
        if len(memory.data.get_all_messages()) == 1:
            return Transition(next_state=self.next_state)

    def after_generation(self, response: str, memory: Memory, tools: Optional[List[ToolAdapter]]) -> Transition:
        return text_handler(response=response, memory=memory, next_state=self.next_state)
