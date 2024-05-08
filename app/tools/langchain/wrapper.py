from typing import Dict, Any, Generic, TypeVar

from langchain_core.tools import BaseTool

from app.core.tools.helper import build_schema

try:
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper
except ImportError as e:
    raise ImportError(
        "Some required modules are missing. "
        "Please install them with 'pip install langchain-community'.") from e

from app.core.tools.adapter import ToolAdapter

from typing import cast


class LangChainToolWrapper:

    @staticmethod
    def create(
            tool: BaseTool,
            exclude_input_from_scratch_pad: bool = False,
            exclude_output_from_scratch_pad: bool = False,
            verbose: bool = False) -> ToolAdapter:
        adapter_class = cast(ToolAdapter, LangChainToolAdapter[tool.InputType, tool.OutputType])
        return adapter_class(
            tool=tool,
            exclude_input_from_scratch_pad=exclude_input_from_scratch_pad,
            exclude_output_from_scratch_pad=exclude_output_from_scratch_pad,
            verbose=verbose
        )


InputType = TypeVar("InputType", bound=BaseTool.InputType)
OutputType = TypeVar("OutputType", bound=BaseTool.OutputType)


class LangChainToolAdapter(ToolAdapter, Generic[InputType, OutputType]):

    def __init__(self,
                 tool: BaseTool,
                 exclude_input_from_scratch_pad: bool = False,
                 exclude_output_from_scratch_pad: bool = False,
                 verbose: bool = False):
        super().__init__()
        self.tool: BaseTool = tool
        self.exclude_input_from_scratch_pad: bool = exclude_input_from_scratch_pad
        self.exclude_output_from_scratch_pad: bool = exclude_output_from_scratch_pad
        self.verbose: bool = verbose

    @property
    def description(self) -> str:
        return self.tool.description

    @property
    def name(self) -> str:
        return self.tool.name

    def run(self, input: InputType) -> OutputType:
        try:
            input = input.dict()
            return self.tool.run(input, return_direct=True, verbose=self.verbose)
        except Exception as e:
            raise ValueError(f"Failed to run tool: {e}")

    def validate(self, input: str) -> InputType:
        model = self.tool.get_input_schema()
        try:
            validated_model = model.validate(input)
            return validated_model
        except Exception as e:
            raise ValueError(f"Failed to validate input: {e}")

    def schema(self) -> Dict[str, Any]:
        input_schema = self.tool.get_input_schema().schema()
        input_schema['title'] = self.name
        return build_schema(self.name, self.description, input_schema)

    def exclude_input_from_scratch_pad(self) -> bool:
        return self.exclude_input_from_scratch_pad

    def exclude_output_from_scratch_pad(self) -> bool:
        return self.exclude_output_from_scratch_pad
