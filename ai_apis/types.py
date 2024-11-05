import re
import json

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

    def replace(self, replacements: dict[str, str], placeholder_format: str = r"\(\(([placeholder])\)\)") -> "Prompt":
        json_string = json.dumps(self.model_dump())
        
        for placeholder, replacement in replacements.items():
            formatted_placeholder = placeholder_format.replace("[placeholder]", placeholder)
            json_string, num_subs = re.subn(formatted_placeholder, replacement, json_string)
            
            if num_subs == 0:
                raise ValueError(f"Missing prompt placeholder: '{formatted_placeholder}'")
        
        print(json_string)
        updated_data = json.loads(json_string, strict=False)
        return Prompt(**updated_data)

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