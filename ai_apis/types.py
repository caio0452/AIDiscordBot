import warnings

from typing import Any, Optional
from pydantic import BaseModel, Field

OpenAIMessage = dict[str, list | str | dict]

class Prompt(BaseModel):
    messages: list[OpenAIMessage] = Field(...)

    @staticmethod
    def system_msg(content: str) -> dict[str, str]:
        return {"role": "system", "content": content}

    @staticmethod
    def user_msg(content: str, image_url: str | None = None) -> dict[str, Any]:
        if image_url:
            return {
                "role": "user",
                "content": [
                    {"type": "text", "text": content},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            }
        else:
            return {"role": "user", "content": content}

    @staticmethod
    def assistant_msg(content: str) -> dict[str, str]:
        return {"role": "assistant", "content": content}

    def replace_in_prompt(self, placeholder: str, target: str) -> "Prompt":
        def replace_in_value(value: Any) -> Any:
            if isinstance(value, str):
                return value.replace(placeholder, target)
            elif isinstance(value, list):
                return [replace_in_value(item) for item in value]
            elif isinstance(value, dict):
                return {k: replace_in_value(v) for k, v in value.items()}
            else:
                return value

        new_messages = [
            replace_in_value(message) for message in self.messages
        ]
        return Prompt(messages=new_messages)

    def to_openai_format(self) -> list[OpenAIMessage]:
        return self.messages

class LLMRequestParams(BaseModel):
    model_name: str
    temperature: float = 0.5
    max_tokens: int = 300
    logit_bias: Optional[dict[str, int]] = {}

    class Config:
        json_encoders = {
            Prompt: lambda p: p.to_openai_format()
        }