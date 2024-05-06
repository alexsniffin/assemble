import enum
from abc import abstractmethod, ABC
from typing import Dict, Any, List


class ContextException(Exception):
    pass


class ContextStrategy(str, enum.Enum):
    TRUNCATE = "truncate"


class BaseHandler(ABC):
    @abstractmethod
    def run(self, context: str) -> str:
        pass


class TruncateHandler(BaseHandler):
    def run(self, context: List[str]) -> List[str]:
        if len(context) == 0:
            raise ContextException("Context cannot be reduced any further.")

        return context[-1:]


class ContextHandler:
    def __init__(self, strategy: ContextStrategy):
        self.strategy_map = {
            ContextStrategy.TRUNCATE: TruncateHandler()
        }
        self.handler = self.strategy_map[strategy]

    def run(self, context: List[str]) -> List[str]:
        return self.handler.run(context)


class ScratchPad:
    """ ScratchPad class for storing and managing session-specific notes. """

    def __init__(self, context_strategy: ContextStrategy):
        self.data: Dict[str, Any] = {}
        self.notes: List[str] = []
        self.context_handler = ContextHandler(context_strategy)

    def set(self, note: str):
        """ Append a note to the scratch pad. """
        self.notes.append(note)

    def clear(self):
        """ Clear all notes from the scratch pad. """
        self.notes = []

    def get(self) -> List[str]:
        """ Retrieve all notes from the scratch pad. """
        return self.notes

    def prompt(self) -> str:
        """ Generate a prompt for the scratch pad. """
        template = "\n{}"
        return template.format("\n- ".join(self.notes))

    def run_context_handler(self):
        self.notes = self.context_handler.run(self.notes)
