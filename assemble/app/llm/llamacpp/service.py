import logging
from typing import Tuple, Dict, List

try:
    from llama_cpp import Llama
except ImportError:
    raise ImportError("Please install LlamaCpp library: pip install llama-cpp-python")

from assemble.app.core.llm.adapter import LLMAdapter

logger = logging.getLogger(__name__)


class LlamaCppService(LLMAdapter):

    def __init__(
            self,
            model_path: str,
            **kwargs):
        self.llama = Llama(
            model_path=model_path,
            **kwargs
        )
        self.model = model_path

    def generate(self, prompt: str, **backend_kwargs) -> Tuple[str, Dict[str, int]]:
        messages = [{"role": "system", "content": prompt}]
        try:
            use_json_model = backend_kwargs.pop("use_json_model", False)
            completion = self.llama.create_chat_completion_openai_v1(
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
        return self.llama.tokenize(bytes(text, 'utf-8'))

    def context_length(self) -> int:
        return self.llama.n_ctx()
