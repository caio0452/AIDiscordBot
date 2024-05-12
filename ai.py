import openai
import json
from typing import Any
from abc import ABC, abstractmethod
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_assistant_message_param import ChatCompletionAssistantMessageParam

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

class OAICompatibleProvider:
    def __init__(self, client: openai.AsyncClient):
        self.client = client

    async def generate_response(self,
                                prompt: list[ChatCompletionMessageParam],
                                model: str,
                                max_tokens: int=300,
                                temperature: float=0.5,
                                logit_bias: dict[str, int] = {}
                                ) -> Choice:
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
                raise RuntimeError(raw_response["error"])
            else:
                raise RuntimeError(f"Provider returned no response choices. Response was {str(raw_response)}")
      

        return raw_response.choices[0]

    @staticmethod
    def system_msg(content: str) -> ChatCompletionSystemMessageParam:
        return {"role": "system", "content": content}

    @staticmethod
    def user_msg(content: str, image_url: str | None = None) -> ChatCompletionUserMessageParam:
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
    def assistant_msg(content: str) -> ChatCompletionAssistantMessageParam:
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

    # To make Pyright happy (list comprehensions do not make Pyright happy)
    def to_openai_format(self) -> list[ChatCompletionMessageParam]:
        ret = []
        for msg in self._dict:
            ret.append(msg)
        return ret

class GPT4Vision():
    def __init__(self, client: openai.AsyncClient):
        self.client = client

    async def describe_image(self, image_url: str, user_context: str) -> Choice:
        initial_msg = OAICompatibleProvider.user_msg("You're an image describer for an AI system interpreting user queries. An user below will make a query. Reply with a description of the image that is enough to answer the user's query. If there's an error message, try to transcribe")
        image_msg = OAICompatibleProvider.user_msg(user_context, image_url=image_url)
        reinforcement_msg = OAICompatibleProvider.system_msg("Reply with just the sufficient, user query related description of the image and nothing else, between brackets like this: [insert decription here]")
        raw_response = await self.client.chat.completions.create(
            messages=[initial_msg, image_msg, reinforcement_msg],
            model="gpt-4-vision-preview",
            max_tokens=300
        )
        return raw_response.choices[0]