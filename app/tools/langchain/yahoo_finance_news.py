from typing import Dict, Any

try:
    from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
except ImportError as e:
    raise ImportError(
        "Some required modules are missing. "
        "Please install them with 'pip install langchain-community'.") from e

try:
    import yfinance as yf
except ImportError as e:
    raise ImportError(
        "Some required modules are missing. "
        "Please install them with 'pip install yfinance'.") from e

try:
    import fake_useragent as fua
except ImportError as e:
    raise ImportError(
        "Some required modules are missing. "
        "Please install them with 'pip install fake-useragent'.") from e

from app.core.tools.adapter import ToolAdapter, ToolSchema, ToolBase


class YahooFinanceNewsInput(ToolBase):
    """
    The input to search for factual information on Wikipedia.
    """
    query: str


class YahooFinanceNewsOutput(ToolBase):
    output: str


class LangchainYahooFinanceNews(ToolAdapter):
    """
    A facade for the Langchain YahooFinanceNewsTool.
    """

    name = "yahoo_finance_news"
    description = (
        "A tool for searching for news on Yahoo Finance. Input should only be the company ticker."
    )

    def __init__(self, top_k: int = 3, verbose: bool = False):
        self.verbose = verbose
        yahoo_finance_news = YahooFinanceNewsTool(
            top_k=top_k
        )
        self.tool = yahoo_finance_news

    def run(self, inputs: YahooFinanceNewsInput) -> YahooFinanceNewsOutput:
        output = self.tool.run(inputs.query, return_direct=True, verbose=self.verbose)
        return YahooFinanceNewsOutput(tool_name=self.name, output=output)

    def validate(self, input: str) -> YahooFinanceNewsInput:
        return YahooFinanceNewsInput.model_validate_json(input)

    def schema(self) -> Dict[str, Any]:
        return ToolSchema(
            tool_name=self.name,
            tool_description=self.description,
            tool_parameters=YahooFinanceNewsInput.model_json_schema()
        ).model_dump()
