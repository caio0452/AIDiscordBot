from typing import Any
from ai_apis import providers
from ai_apis.client import LLMClient
from bot_workflow.vector_db import VectorDatabase
from ai_apis.types import LLMRequestParams, Prompt
from bot_workflow.personality_loader import Personality
from bot_workflow.types import AIBotData, MemorizedMessageHistory

import json
import discord
import datetime

MAIN_CLIENT_NAME = "PERSONALITY"
KNOWLEDGE_EXTRACTOR_NAME = "KAMI_CHAN_KNOWLEDGE_EXTRACTOR"
PERSONALITY_REWRITER_NAME = "KAMI_CHAN_PERSONALITY_REWRITER"
IMAGE_VIEWER_NAME = "KAMI_CHAN_IMAGE_VIEWER"

class CustomBotData(AIBotData):
    def __init__(self,
                 name: str,
                 vector_db: VectorDatabase,
                 personality: Personality,
                 provider_store: providers.ProviderDataStore,
                 discord_bot_id: int
                ):
        super().__init__(name, MemorizedMessageHistory())
        self.provider_store = provider_store
        self.discord_bot_id = discord_bot_id
        self.vector_db = vector_db
        self.recent_history = MemorizedMessageHistory()
        self.RECENT_MEMORY_LENGTH = 5
        self.personality = personality

class DiscordBotResponse:
    def __init__(self, bot_data: CustomBotData, verbose: bool=False):
        self.verbose = verbose
        self.bot_data = bot_data
        self.verbose_log = ""
        self.clients: dict[str, LLMClient] = {}

        for k, v in bot_data.provider_store.providers.items():
            self.clients[k] = LLMClient.from_provider(v)

    def log_verbose(self, text: str, *, category: str|None = None):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if category:
            self.verbose_log += f"[{current_time}] --- {category} ---\n{text}\n"
        else:
            self.verbose_log += f"[{current_time}] {text}\n"

    async def create_or_fallback(self, message: discord.Message, model_names: list[str]) -> str:
        all_errors = []
        full_prompt = await self.build_full_prompt(
            self.bot_data.recent_history.without_dupe_ending_user_msgs(), 
            message
        )
        
        for model_name in model_names:
            try:
                response = await self.clients[MAIN_CLIENT_NAME].send_request(
                    prompt=Prompt(messages=full_prompt),
                    params=LLMRequestParams(
                        model_name=model_name,
                        max_tokens=2000,
                        temperature=0
                    )
                )
                response_txt = response.message.content
                self.log_verbose(response_txt, category="PERSONALITY-LESS MESSAGE")
                self.log_verbose(f"Length (chars): {len(response_txt)}")
                print(f"Rewriting {response_txt}")
                personality_rewrite = await self.personality_rewrite(response_txt)
                self.log_verbose(personality_rewrite, category="IN-CHARACTER REWRITE")
                self.log_verbose(str(response), category="RAW API RESPONSE")
                return personality_rewrite
            except Exception as e:
                self.log_verbose(f"Model {model_name} failed with error: {e}", category="MODEL FAILURE")
                all_errors.append(e)
    
        raise RuntimeError("Could not generate response and all fallbacks failed") from Exception(all_errors)

    async def create(self, message: discord.Message) -> str:
        return await self.create_or_fallback(message, ["google/gemini-pro-1.5-exp", "openai/gpt-4o-mini", "meta-llama/llama-3.1-405b-instruct"])

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
        print(f"Returned: {response.message.content}")
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
        return response.message.content

    async def info_select(self, user_query: str) -> str | None:
        NAME = "INFO_SELECT"
        user_prompt_str = ""
        knowledge_list = await self.bot_data.vector_db.search(user_query, 5, "knowledge")

        if len(knowledge_list) == 0:
            return None

        for knowledge in knowledge_list:
            user_prompt_str += "INFO: \n" + str(knowledge) # TODO: what's the type of this?

        user_prompt_str += "QUERY: " + user_query
        prompt = self.bot_data.personality.prompts[NAME] \
            .replace({
                "user_query": user_prompt_str
            })

        response = await self.send_llm_request(
            name=NAME,
            prompt=prompt
        )
        return response.message.content

    async def describe_image_if_present(self, message) -> str | None:
        if len(message.attachments) == 1:
            if message.channel.nsfw:
                await message.reply(":X: I can't see attachments in NSFW channels!")
                return None
            attachment = message.attachments[0]
            if attachment.content_type.startswith("image/"):
                await message.add_reaction("👀")
                # Todo: only last message is possibly not enough context
                response = await self.clients[IMAGE_VIEWER_NAME].send_request(
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

    async def build_full_prompt(self, memory_snapshot: MemorizedMessageHistory, original_msg: discord.Message) -> list[Any]:
        now_str = datetime.datetime.now().strftime("%B %d, %H:%M:%S")
        user_query = await self.user_query_rephrase()

        # TODO: the "Prompt" class is weirdly used here, but not worth looking too much into as of this will be replaced by a JSON file eventually
        bot_prompt = self.bot_data.personality.prompts[MAIN_CLIENT_NAME]
        system_prompt_str = bot_prompt.replace(
            {
                "nick": memory_snapshot.as_list()[-1].nick,
                "now": now_str
            }).messages[0]["content"]

        if not isinstance(system_prompt_str, str):
            raise RuntimeError("System prompt must be plain text")

        knowledge = await self.info_select(user_query)

        if knowledge is not None:
            system_prompt_str += f"\n[INFO FROM KNOWLEDGE DB]:\n{knowledge}\n"
            self.log_verbose(knowledge, category="INFO FROM KNOWLEDGE DB")
        else:
            self.log_verbose("The knowledge database has nothing relevant", category="INFO FROM KNOWLEDGE DB")

        prompt: list[Any] = [
            Prompt.system_msg(system_prompt_str)
        ]
        old_messages_str = ""
        # for old_message in await self.bot_data.vector_db_conn.query_relevant_messages(original_msg.content):
        #     old_messages_str += f"[{old_message.sent.isoformat()} by {old_message.nick}] {old_message.text}"

        system_prompt_str += \
        f"""
        [[OLD MESSAGES IN YOUR MEMORY]]:\n{old_messages_str}\n\n
        [[RECENT CONVERSATION HISTORY]]:\n{knowledge}
        """

        for memorized_message in memory_snapshot.as_list():
            if memorized_message.is_bot:
                prompt.append(Prompt.assistant_msg(memorized_message.text))
            else:
                prompt.append(Prompt.user_msg(memorized_message.text))

        img_desc = await self.describe_image_if_present(original_msg)
        if img_desc:
            prompt.append(Prompt.user_msg(img_desc))

        self.log_verbose(f"--- FULL PROMPT ---\n{json.dumps(prompt, indent=4)}\n")
        return prompt

    async def send_llm_request(self, *, name: str, prompt: Prompt):
        params = self.bot_data.personality.request_params[name]
        provider: providers.ProviderData = self.bot_data.personality.providers[name]
        client: LLMClient = LLMClient.from_provider(provider)

        return await client.send_request(prompt=prompt, params=params) 