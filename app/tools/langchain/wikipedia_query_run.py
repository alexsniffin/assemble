from typing import Dict, Any

try:
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper
except ImportError as e:
    raise ImportError(
        "Some required modules are missing. "
        "Please install them with 'pip install langchain-community'.") from e

try:
    import wikipedia
except ImportError as e:
    raise ImportError(
        "Some required modules are missing. "
        "Please install it with 'pip install wikipedia'.") from e

from app.core.tools.adapter import ToolAdapter, ToolSchema, ToolBase


class WikipediaQueryInput(ToolBase):
    """
    The input to search for factual information on Wikipedia.
    """
    query: str


class WikipediaQueryOutput(ToolBase):
    output: str


class LangchainWikipediaQueryRun(ToolAdapter):
    """
    A facade for the Langchain Wikipedia tool.
    """

    name = "wikipedia_query_run"
    description = (
        "A tool for querying the Wikipedia API."
    )

    def __init__(self,
                 top_k_results: int = 3,
                 doc_content_chars_max: int = 1024,
                 verbose: bool = False,
                 exclude_input_from_scratch_pad: bool = False,
                 exclude_output_from_scratch_pad: bool = False):
        self.verbose = verbose
        api_wrapper = WikipediaAPIWrapper(
            top_k_results=top_k_results,
            doc_content_chars_max=doc_content_chars_max,
            load_all_available_meta=True
        )
        self.tool = WikipediaQueryRun(api_wrapper=api_wrapper)
        self.exclude_input_from_scratch_pad = exclude_input_from_scratch_pad
        self.exclude_output_from_scratch_pad = exclude_output_from_scratch_pad

    def run(self, inputs: WikipediaQueryInput) -> WikipediaQueryOutput:
        output = self.tool.run(inputs.query, return_direct=True, verbose=self.verbose)
        return WikipediaQueryOutput(tool_name=self.name, output=output)

    def validate(self, input: str) -> WikipediaQueryInput:
        return WikipediaQueryInput.model_validate_json(input)

    def schema(self) -> Dict[str, Any]:
        return ToolSchema(
            tool_name=self.name,
            tool_description=self.description,
            tool_parameters=WikipediaQueryInput.model_json_schema()
        ).model_dump()

    def exclude_input_from_scratch_pad(self) -> bool:
        return self.exclude_input_from_scratch_pad

    def exclude_output_from_scratch_pad(self) -> bool:
        return self.exclude_output_from_scratch_pad
