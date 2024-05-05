from typing import Tuple, Dict, List

from app.llm.adapter import LLMAdapter


class Generator:
    def __init__(self, service: LLMAdapter, token_limit_buffer: int = 512, **backend_kwargs):
        self.service = service
        self.backend_kwargs = backend_kwargs
        self.token_limit_buffer = token_limit_buffer

    def generate(self, prompt: str) -> Tuple[str, Dict[str, int]]:
        return self.service.generate(prompt, **self.backend_kwargs)

    def is_context_limit(self, prompt: str) -> bool:
        tokenized = self.service.tokenize(prompt)
        return len(tokenized) + self.token_limit_buffer > self.service.context_length()
