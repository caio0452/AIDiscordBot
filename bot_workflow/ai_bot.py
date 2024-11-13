from ai_apis import providers
from ai_apis.client import LLMClient
from bot_workflow.vector_db import VectorDatabase
from bot_workflow.knowledge import KnowledgeIndex
from ai_apis.types import LLMRequestParams, Prompt
from bot_workflow.personality_loader import Personality
from bot_workflow.types import AIBotData, MemorizedMessageHistory

import re
import discord
import datetime
import traceback

class CustomBotData(AIBotData):
    def __init__(self,
                 *,
                 name: str,
                 vector_db: VectorDatabase,
                 personality: Personality,
                 provider_store: providers.ProviderDataStore,
                 knowledge: KnowledgeIndex,
                 discord_bot_id: int,
                ):
        super().__init__(name, MemorizedMessageHistory())
        self.personality = personality
        self.provider_store = provider_store
        self.discord_bot_id = discord_bot_id
        self.vector_db = vector_db
        self.recent_history = MemorizedMessageHistory()
        self.knowledge = knowledge
        self.RECENT_MEMORY_LENGTH = personality.recent_message_history_length

class ResponseLogger:
    def __init__(self):
        self.text = ""

    def verbose(self, text: str, *, category: str | None = None):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if category:
            self.text += f"[{current_time}] --- {category} ---\n{text}\n"
        else:
            self.text += f"[{current_time}] {text}\n"

class DiscordBotResponse:
    def __init__(self, bot_data: CustomBotData, verbose: bool=False):
        self.verbose = verbose
        self.bot_data = bot_data
        self.logger: ResponseLogger = ResponseLogger()
        self.clients: dict[str, LLMClient] = {}

        for k, v in bot_data.provider_store.providers.items():
            self.clients[k] = LLMClient.from_provider(v)

    # TODO: clean this method up
    async def create_or_fallback(self, message: discord.Message, model_names: list[str]) -> str:
        MAIN_CLIENT_NAME = "PERSONALITY"
        full_prompt = await self.build_full_prompt(
            self.bot_data.recent_history.without_dupe_ending_user_msgs(), 
            message
        )
        for model_name in model_names:
            try:
                response = await self.clients[MAIN_CLIENT_NAME].send_request(
                    prompt=full_prompt,
                    params=LLMRequestParams(
                        model_name=model_name,
                        max_tokens=2000,
                        temperature=0
                    )
                )
                self.logger.verbose(f"Pre-rewrite response: {response}", category="PERSONALITY RESPONSE")
                personality_rewrite = await self.personality_rewrite(response.message.content)
                answer_with_replacements = personality_rewrite
                for k, v in self.bot_data.personality.regex_replacements.items():
                    answer_with_replacements = re.sub(k, v, answer_with_replacements)
                return answer_with_replacements
            except Exception as e:
                traceback.print_exc()
                self.logger.verbose(f"Model {model_name} failed with error: {e}", category="MODEL FAILURE")
        
        raise RuntimeError("Could not generate response and all fallbacks failed")
    
    async def personality_rewrite(self, message: str) -> str:
        NAME = "PERSONALITY_REWRITE"
        name_prompt = self.bot_data.personality.prompts[NAME]
        prompt = name_prompt.replace({
            "message": message
        })
        response = await self.send_llm_request(
            name=NAME,
            prompt=prompt
        )  
        self.logger.verbose(f"Prompt: {prompt}\nResponse: {response}", category=NAME)
        return response.message.content

    async def user_query_rephrase(self) -> str:
        NAME = "USER_QUERY_REPHRASE"
        user_prompt_str = "\n".join(
            [memorized_message.text for memorized_message in self.bot_data.recent_history.as_list()]
        )
        last_user = self.bot_data.recent_history.as_list()[-1].nick
        prompt = self.bot_data.personality.prompts[NAME].replace({
            "user_query": user_prompt_str, 
            "last_user": last_user
        })
        response = await self.send_llm_request(
            name=NAME,
            prompt=prompt
        )
        self.logger.verbose(f"Prompt: {prompt}\nResponse: {response}", category=NAME)
        return response.message.content

    async def info_select(self, user_query: str) -> str | None:
        NAME = "INFO_SELECT"
        user_prompt_str = ""
        knowledge_list = await self.bot_data.knowledge.retrieve(user_query)

        if len(knowledge_list) == 0:
            return None

        for knowledge in knowledge_list:
            text_content: str = knowledge["text"]
            user_prompt_str += f"INFO:n{text_content}"

        user_prompt_str += "QUERY: " + user_query
        prompt = self.bot_data.personality.prompts[NAME] \
            .replace({
                "user_query": user_prompt_str
            })
        response = await self.send_llm_request(
            name=NAME,
            prompt=prompt
        )
        self.logger.verbose(f"Prompt: {prompt}\nResponse: {response}", category=NAME)
        return response.message.content

    # TODO: clean this method up
    async def describe_image_if_present(self, message) -> str | None:
        NAME = "IMAGE_VIEW"
        if len(message.attachments) == 1:
            if message.channel.nsfw:
                await message.reply(":X: I can't see attachments in NSFW channels!")
                return None
            attachment = message.attachments[0]
            if attachment.content_type.startswith("image/"):
                await message.add_reaction("👀")
                # Todo: only last message is possibly not enough context
                response = await self.clients[NAME].send_request(
                    prompt=Prompt(
                                messages=[
                                    Prompt.user_msg(
                                        content=f"Describe the image in a sufficient way to answer the following query: '{message.content}'" \
                                        "If the query is empty, just describe the image. ",
                                        image_url=attachment.url
                                    )
                                ]
                            ),
                   params=LLMRequestParams(
                         model_name="openai/gpt-4o"
                    )
                )
                return response.message.content

    async def build_full_prompt(self, memory_snapshot: MemorizedMessageHistory, original_msg: discord.Message) -> Prompt:
        NAME = "PERSONALITY"
        now_str = datetime.datetime.now().strftime("%B %d, %H:%M:%S")
        user_query = await self.user_query_rephrase()
        knowledge = await self.info_select(user_query)
        old_memories: str = "" # TODO: implement
        full_prompt: Prompt = self.bot_data.personality.prompts[NAME]

        if knowledge is not None:
            knowledge_str = f"\n[INFO FROM KNOWLEDGE DB]:\n{knowledge}\n"
            self.logger.verbose(knowledge, category="INFO FROM KNOWLEDGE DB")
        else:
            knowledge_str = ""
            self.logger.verbose("The knowledge database has nothing relevant", category="INFO FROM KNOWLEDGE DB")

        for memorized_message in memory_snapshot.as_list():
            if memorized_message.is_bot:
                full_prompt.append(Prompt.assistant_msg(memorized_message.text))
            else:
                full_prompt.append(Prompt.user_msg(memorized_message.text))

        img_desc = await self.describe_image_if_present(original_msg)
        if img_desc:
            full_prompt.append(Prompt.user_msg(img_desc))

        full_prompt.replace({
            "now": now_str,
            "nick": original_msg.author.display_name,
            "knowledge": knowledge_str,
            "old_memories": old_memories
        })

        self.logger.verbose(f"FULL PROMPT: {full_prompt}", category="FULL PROMPT")
        return full_prompt

    async def send_llm_request(self, *, name: str, prompt: Prompt):
        params = self.bot_data.personality.request_params[name]
        provider: providers.ProviderData = self.bot_data.personality.providers[name]
        client: LLMClient = LLMClient.from_provider(provider)

        return await client.send_request(prompt=prompt, params=params) 