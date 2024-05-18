import openai
import json
from typing import Any, List
from abc import ABC, abstractmethod
from openai.types import CompletionChoice

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

async def _text_to_vector(self, openai_client: openai.AsyncOpenAI, texts: List[str]) -> List[List[float]]:
    response = await openai_client.embeddings.create(
        input=texts,
        model="text-embedding-3-large"
    )
    vectors = [e.embedding for e in response.data]
    return vectors
    
class OAICompatibleProvider:
    def __init__(self, client: openai.AsyncClient):
        self.client = client

    async def vectorize(self, text: str, model="text-embedding-3-large") -> List[float]:
        model_in = [text]
        response = await self.client.embeddings.create(
            input=model_in,
            model=model
        )
        return response.data[0].embedding

    async def vectorize_many(self, texts: List[str], model="text-embedding-3-large") -> List[List[float]]:
        model_in = texts
        response = await self.client.embeddings.create(
            input=model_in,
            model=model
        )
        return [e.embedding for e in response.data]

    async def generate_response(self,
                                prompt: list[dict[str, str]],
                                model: str,
                                max_tokens: int=300,
                                temperature: float=0.5,
                                logit_bias: dict[str, int] = {}
                                ) -> openai.types.CompletionChoice:
        raw_response = await self.client.chat.completions.create(
            messages=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            logit_bias=logit_bias
        )

        if raw_response.choices is None or len(raw_response.choices) == 0:
            resp_json = json.loads(raw_response.to_json())
            if "error" in resp_json and resp_json["error"]: # For OpenRouter compat
                raise RuntimeError(resp_json["error"])
            else:
                raise RuntimeError(f"Provider returned no response choices. Response was {str(raw_response)}")
        else:
            return raw_response.choices[0]
      
    async def describe_image(self, image_url: str, user_context: str) -> CompletionChoice:
        initial_msg = OAICompatibleProvider.user_msg(f"Describe the image. {user_context}", image_url=image_url)
        raw_response = await self.client.chat.completions.create(
            messages=[initial_msg],
            model="anthropic/claude-3-haiku",
            max_tokens=1000
        )
        return raw_response.choices[0]
        
    @staticmethod
    def system_msg(content: str) -> dict[str, str]:
        return {"role": "system", "content": content}

    @staticmethod
    def user_msg(content: str, image_url: str | None = None) -> dict[str, str]:
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

class Prompt:
    def __init__(self, messages: list[dict[str, str]]):
        self._dict = messages

    def replace(self, placeholder: str, target: str) -> "Prompt":
        found_placeholder = False
        new_prompts = []
        for msg in self._dict:
            new_prompt_dict = {}
            for key, value in msg.items():
                new_value = value.replace(placeholder, target)
                if new_value != value:
                    found_placeholder = True
                new_prompt_dict[key] = new_value
            new_prompts.append(new_prompt_dict)

        if not found_placeholder:
            raise ValueError(f"Placeholder '{placeholder}' not found")

        return Prompt(new_prompts)

    def to_openai_format(self) -> dict[str, str]:
        return self._dict