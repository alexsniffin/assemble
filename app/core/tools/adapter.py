from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Dict, TypedDict, Any

from pydantic import BaseModel, Field


class ToolSchema(BaseModel):
    tool_name: str = Field(..., description="The name of the tool.")
    tool_description: str = Field(..., description="The description of the tool.")
    tool_parameters: Dict[str, Any] = Field(..., description="The parameters of the tool.")


class ToolBase(BaseModel):
    tool_name: str = Field(..., description="The name of the tool you choose to use.")


InputType = TypeVar("InputType", bound=ToolBase)
OutputType = TypeVar("OutputType", bound=ToolBase)


class ToolAdapter(ABC, Generic[InputType, OutputType]):
    """ Abstract base class for tool adapters, defining a standard interface for tools. """

    @abstractmethod
    def run(self, inputs: InputType) -> OutputType:
        pass

    @abstractmethod
    def validate(self, input: str) -> InputType:
        pass

    @abstractmethod
    def schema(self) -> ToolSchema:
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
    def exclude_from_scratch_pad(self) -> bool:
        """ Return whether the tool should be excluded from the scratch pad. """
        pass
