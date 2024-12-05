from abc import ABC, abstractmethod
from typing import Tuple, Dict, List


class LLMAdapter(ABC):

    @abstractmethod
    def generate(self, prompt: str, **backend_kwargs) -> Tuple[str, Dict[str, int]]:
        pass

    @abstractmethod
    def tokenize(self, text: str) -> List[int]:
        pass

    @abstractmethod
    def context_length(self) -> int:
        pass
