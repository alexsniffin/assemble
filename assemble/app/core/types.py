from pydantic import BaseModel


class Usage(BaseModel):
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
