import time

from core.ai_apis import providers
from abc import ABC, abstractmethod
from core.bot_workflow.ai_bot import Prompt, LLMClient, CustomBotData
from core.bot_workflow.response_logs import ResponseLogger

class ResponseStep(ABC):
    def __init__(self):
        self.finished = False
        self.elapsed_ms: float | None = None
        self.logger = ResponseLogger()

    async def _llm_request(self, *, name: str, prompt: Prompt):
        params = self.bot_data.profile.request_params[name]
        provider: providers.ProviderData = self.bot_data.profile.providers[name]
        client: LLMClient = LLMClient.from_provider(provider)
        return await client.send_request(prompt=prompt, params=params)
    
    async def execute(self, bot_data: CustomBotData, message: str) -> str | None:
        self.bot_data = bot_data
        self.message = message
        start = time.perf_counter()
        ret = await self._run()
        end = time.perf_counter()
        self.elapsed_ms = 1000 * (end - start)
        self.finished = True
        return ret

    @abstractmethod
    async def _run(self) -> str | None:
        raise NotImplementedError("_run() for ResponseStep")
    
class PersonalityRewriteStep(ResponseStep):
    async def _run(self):
        NAME = "PERSONALITY_REWRITE"
        name_prompt = self.bot_data.profile.get_prompt(NAME)
        prompt = name_prompt.replace({
            "message": self.message
        })
        response = await self._llm_request(
            name=NAME,
            prompt=prompt
        ) 
        self.logger.verbose(f"PROMPT: {prompt}", category="PERSONALITY REWRITE") 
        return response.message.content
    
class UserQueryRephraseStep(ResponseStep):
    async def _run(self):
        NAME = "USER_QUERY_REPHRASE"
        recent_history_list = self.bot_data.recent_history.backing_history.as_list()
        user_prompt_str = "\n".join(
            [memorized_message.text for memorized_message in recent_history_list]
        )
        last_user = recent_history_list[-1].nick
        prompt = self.bot_data.profile.get_prompt(NAME).replace({
            "user_query": user_prompt_str, 
            "last_user": last_user
        })
        response = await self._llm_request(
            name=NAME,
            prompt=prompt
        )
        self.logger.verbose(f"Prompt: {prompt}\nResponse: {response}", category=NAME)
        return response.message.content
    
class RelevantInfoSelectStep(ResponseStep):
    def __init__(self, user_query: str):
        super().__init__()
        self.user_query = user_query

    async def _run(self):
        NAME = "INFO_SELECT"
        available_info = ""
        hits_list = await self.bot_data.knowledge.retrieve(self.user_query)

        if len(hits_list) == 0:
            return None

        for hits in hits_list:
            for hit in hits:
                available_info += hit["text"] + "\n"

        prompt = self.bot_data.profile.get_prompt(NAME) \
            .replace({
                "user_query": self.user_query,
                "available_info": available_info
            })
        response = await self._llm_request(
            name=NAME,
            prompt=prompt
        )
        self.logger.verbose(f"Prompt: {prompt}\nResponse: {response}", category=NAME)
        return response.message.content