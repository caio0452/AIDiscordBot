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

    def replace(self, placeholder: str, target: str) -> "Prompt":
        found_placeholder = False
        new_prompts = []
        for msg in self.messages:
            new_prompt_dict = {}
            for key, value in msg.items():
                # TODO: value is not necessarily a string. It could be a dict. Implement recursive replacing
                if isinstance(value, str):
                    new_value = value.replace(placeholder, target)
                else:
                    warnings.warn(f"Placeholders in complex (non-string) messages are not supported yet, will not scan: {value}")
                    continue

                if new_value != value:
                    found_placeholder = True
                new_prompt_dict[key] = new_value
            new_prompts.append(new_prompt_dict)

        if not found_placeholder:
            raise ValueError(f"Placeholder '{placeholder}' not found")

        return Prompt(messages=new_prompts)

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