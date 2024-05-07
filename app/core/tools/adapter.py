from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Dict, Any

from pydantic import BaseModel


class ToolInput(BaseModel):
    name: str
    parameters: Dict[str, Any]


class ToolDetails(BaseModel):
    name: str
    description: str


InputType = TypeVar("InputType", bound=BaseModel)
OutputType = TypeVar("OutputType", bound=BaseModel)


class ToolAdapter(ABC, Generic[InputType, OutputType]):

    @abstractmethod
    def run(self, input: InputType) -> OutputType:
        pass

    @abstractmethod
    def validate(self, input: str) -> InputType:
        pass

    @abstractmethod
    def schema(self) -> Dict[str, Any]:
        """ Return the schema for input/output validation. """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """ Return the name of the tool. """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """ Return a description of what the tool does. """
        pass

    @property
    @abstractmethod
    def exclude_output_from_scratch_pad(self) -> bool:
        """ Return whether the tool should be excluded from the scratch pad. """
        pass

    @property
    @abstractmethod
    def exclude_input_from_scratch_pad(self) -> bool:
        """ Return whether the tool should be excluded from the scratch pad. """
        pass
