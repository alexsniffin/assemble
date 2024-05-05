from dataclasses import dataclass
from typing import Optional


@dataclass
class Query:
    goal: str
    initial_state: Optional[str] = None
    from_caller: str = "user"
