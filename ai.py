import json
import openai

from providers import Provider
from typing import Any, Optional
from abc import ABC, abstractmethod
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
                new_value = value.replace(placeholder, target)
                if new_value != value:
                    found_placeholder = True
                new_prompt_dict[key] = new_value
            new_prompts.append(new_prompt_dict)

        if not found_placeholder:
            raise ValueError(f"Placeholder '{placeholder}' not found")

        return Prompt(messages=new_prompts)

    def to_openai_format(self) -> list[OpenAIMessage]:
        return self.messages

class LLMRequest(BaseModel):
    prompt: Prompt 
    model_name: str
    temperature: float = 0.5
    max_tokens: int = 300
    logit_bias: Optional[dict[str, int]] = {}

    class Config:
        json_encoders = {
            Prompt: lambda p: p.to_openai_format()
        }

class ContentModerator(ABC):
    @abstractmethod
    async def is_flagged(self, input: Any) -> bool:
        raise NotImplementedError("is_flagged")

class OpenAIModerator(ContentModerator):
    def __init__(self, client: openai.AsyncClient):
        self.client = client

    async def is_flagged(self, input) -> Any:
        response = await self.client.moderations.create(
            input=input
        )
        return response.results[0].flagged

# TODO: should be provider (e.g. OpenAI) agnostic
class EmbeddingsClient:
    def __init__(self, provider: Provider):
        self.client = openai.AsyncOpenAI(
            api_key=provider.api_key, 
            base_url=provider.api_base
        )

    async def vectorize(self, input: str | list[str], model="text-embedding-3-large") -> list[float] | list[list[float]]:
        if isinstance(input, str):
            response = await self.client.embeddings.create(
                input=input,
                model=model
            )
            return response.data[0].embedding
        elif isinstance(input, list):
            response = await self.client.embeddings.create(
                input=input,
                model=model
            )
            return [e.embedding for e in response.data]

class SyncEmbeddingsClient:
    def __init__(self, provider: Provider):
        self.client = openai.OpenAI(
            api_key=provider.api_key, 
            base_url=provider.api_base
        )

    def vectorize(self, input: str | list[str], model="text-embedding-3-large") -> list[float] | list[list[float]]:
        if isinstance(input, str):
            response = self.client.embeddings.create(
                input=input,
                model=model
            )
            return response.data[0].embedding
        elif isinstance(input, list):
            response = self.client.embeddings.create(
                input=input,
                model=model
            )
            return [e.embedding for e in response.data]

class LLMProvider:
    def __init__(self, client: openai.AsyncClient):
        self.client = client

    @classmethod
    def from_openai_client(cls, client: openai.AsyncClient):
        return cls(client)

    async def send_request(self, request: LLMRequest) -> openai.types.CompletionChoice:
        raw_response = await self.client.chat.completions.create(
            messages=request.prompt.to_openai_format(),
            model=request.model_name,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            logit_bias=request.logit_bias
        )

        if raw_response.choices is None or len(raw_response.choices) == 0:
            resp_json = json.loads(raw_response.to_json())
            if "error" in resp_json and resp_json["error"]:
                raise RuntimeError(resp_json["error"])
            else:
                raise RuntimeError(f"Provider returned no response choices. Response was {str(raw_response)}")
        else:
            return raw_response.choices[0]
        
