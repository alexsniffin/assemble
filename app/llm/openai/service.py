import logging
from typing import Tuple, Dict, List

try:
    import tiktoken
except ImportError:
    raise ImportError("Please install tiktoken library: pip install tiktoken")

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Please install OpenAI library: pip install openai")

from app.core.llm.adapter import LLMAdapter

logger = logging.getLogger(__name__)


class OpenAIService(LLMAdapter):

    def __init__(
            self,
            api_key: str,
            model: str = "gpt-3.5-turbo-0125",
            encoding_type: str = "cl100k_base",
            context_window_length: int = 4094):
        self.openai = OpenAI(api_key=api_key)
        self.model = model
        self.encoding_type = encoding_type
        self._model_context_windows = {
            "gpt-4-turbo": 128000,
            "gpt-4-turbo-2024-04-09": 128000,
            "gpt-4-turbo-preview": 128000,
            "gpt-4-0125-preview": 128000,
            "gpt-4-1106-preview": 128000,
            "gpt-4-vision-preview": 128000,
            "gpt-4-1106-vision-preview": 128000,
            "gpt-4": 8192,
            "gpt-4-0613": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-32k-0613": 32768,
            "gpt-3.5-turbo-0125": 16385,
            "gpt-3.5-turbo": 16385,
            "gpt-3.5-turbo-1106": 16385,
            "gpt-3.5-turbo-instruct": 4096,
            "gpt-3.5-turbo-16k": 16385,
            "gpt-3.5-turbo-0613": 4096,
            "gpt-3.5-turbo-16k-0613": 16385
        }
        if self.model not in self._model_context_windows:
            self._model_context_windows[self.model] = context_window_length

    def generate(self, prompt: str, **backend_kwargs) -> Tuple[str, Dict[str, int]]:
        messages = [{"role": "system", "content": prompt}]
        try:
            use_json_model = backend_kwargs.pop("use_json_model", False)
            completion = self.openai.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"} if use_json_model else {"type": "text"},
                **backend_kwargs
            )

            response_content = completion.choices[0].message.content
            logger.debug(f"LLM response: {response_content}")

            return response_content, {
                "completion_tokens": completion.usage.completion_tokens,
                "prompt_tokens": completion.usage.prompt_tokens,
                "total_tokens": completion.usage.total_tokens
            }
        except Exception as e:
            logger.error(f"Failed to generate LLM response: {e}")
            raise e

    def tokenize(self, text: str) -> List[int]:
        enc = tiktoken.get_encoding(self.encoding_type)
        return enc.encode(text)

    def context_length(self) -> int:
        return self._model_context_windows[self.model]
