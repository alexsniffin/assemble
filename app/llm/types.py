from dataclasses import dataclass


@dataclass
class Usage:
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
