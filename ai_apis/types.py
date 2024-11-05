import re
import json
import traceback

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

    def replace(self, replacements: dict[str, str], placeholder_format: str = "(([placeholder]))") -> "Prompt":
        data_dict = self.model_dump()
        
        def replace_all_in_dict(dict_data, old_str, new_str):
            if isinstance(dict_data, dict):
                return {k: replace_all_in_dict(v, old_str, new_str) for k, v in dict_data.items()}
            elif isinstance(dict_data, str):
                return dict_data.replace(old_str, new_str)
            else:
                return dict_data 

        json_string = json.dumps(data_dict)
        for placeholder in replacements:
            if placeholder not in json_string:
                raise ValueError(f"Missing prompt placeholder: '{placeholder}'")

        result_dict = data_dict
        for placeholder, replacement in replacements.items():
            formatted_placeholder = placeholder_format.replace("[placeholder]", placeholder)
            result_dict = replace_all_in_dict(result_dict, formatted_placeholder, replacement)
        
        return self.__class__.model_validate(result_dict)

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