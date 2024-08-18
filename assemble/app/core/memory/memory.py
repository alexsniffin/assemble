import logging
from dataclasses import dataclass
from typing import Any, List

from assemble.app.core.memory.scratch_pad import ContextStrategy, ScratchPad

logger = logging.getLogger(__name__)


@dataclass
class Message:
    name: str
    content: str


class DataStore:
    _messages_key = "messages"

    def __init__(self):
        messages: List[Message] = []
        self.data = {
            self._messages_key: messages,
        }

    def exists(self, key) -> bool:
        return key in self.data

    def get(self, key) -> Any:
        return self.data.get(key, None)

    def set(self, key, value):
        self.data[key] = value

    def pop(self, key) -> Any:
        return self.data.pop(key, None)

    def remove(self, key):
        self.data.pop(key, None)

    def get_current_message(self) -> Message:
        return self.data[self._messages_key][-1]

    def get_all_messages(self) -> List[Message]:
        return self.data[self._messages_key]

    def add_message(self, message: Message):
        self.data[self._messages_key].append(message)

    def clear_messages(self):
        self.data[self._messages_key] = []


class Memory:
    """ Memory class for storing and managing session-specific data and notes. """

    def __init__(self, context_strategy: ContextStrategy = ContextStrategy.TRUNCATE):
        self.data: DataStore = DataStore()
        self.scratch_pad: ScratchPad = ScratchPad(context_strategy)

    def reset_data(self):
        self.data = DataStore()

    def reset_scratch_pad(self):
        self.scratch_pad.clear()

    def run_context_handlers(self):
        self.scratch_pad.run_context_handler()
