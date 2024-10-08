from . import prompts
from typing import Any
from datetime import datetime
from ai import LLMRequest, Prompt, LLMProvider
from vector_db import VectorDatabase
from ai_bot import AIBotData, BotMemory

import datetime
import providers
import random, discord, openai

#
# TODO: This is a mess that has to be replaced with something data-driven
#
MAIN_CLIENT_NAME = "DEFAULT"
KNOWLEDGE_EXTRACTOR_NAME = "KAMI_CHAN_KNOWLEDGE_EXTRACTOR"
PERSONALITY_REWRITER_NAME = "KAMI_CHAN_PERSONALITY_REWRITER"
IMAGE_VIEWER_NAME = "KAMI_CHAN_IMAGE_VIEWER"

class KamiChan(AIBotData):
    def __init__(self,
                 name: str,
                 vector_db: VectorDatabase,
                 provider_store: providers.ProviderStore,
                 discord_bot_id: int):
        super().__init__(name, BotMemory())
        self.provider_store = provider_store
        self.clients: dict[str, Any] = {}
        self._create_clients()
        self.discord_bot_id = discord_bot_id
        self.vector_db = vector_db
        self.memory: BotMemory = BotMemory()
        self.RECENT_MEMORY_LENGTH = 5
        
    def _create_clients(self):
        required_provider_names = [
           MAIN_CLIENT_NAME, KNOWLEDGE_EXTRACTOR_NAME, PERSONALITY_REWRITER_NAME, IMAGE_VIEWER_NAME
        ]
        for name in required_provider_names:
            provider = self.provider_store.get_provider_by_name(name)
            client = LLMProvider(openai.AsyncOpenAI(
                api_key=provider.api_key, base_url=provider.api_base))
            self.clients[name] = client

    async def sanitize_msg(self, message: discord.Message) -> str:
        new_content = message.content
        for mention in message.mentions:
            new_content = new_content.replace(f'<@{mention.id}>', f'@{mention.name}')
        return await self.sanitize_str(new_content)

    async def sanitize_str(self, message: str) -> str:
        sanitized = message
        if sanitized.startswith(f"<@{self.discord_bot_id}>"):
            sanitized = sanitized.replace(f"<@{self.discord_bot_id}>", "", 1)
        return sanitized

    class Vocabulary:
        EMOJI_NO = "<:Paperno:1022991562810077274>"
        EMOJI_DESPAIR = "<a:notlikepaper:1165467302578360401>"
        EMOJI_UWU = "<:paperUwU:1018366709658308688>"
        EMOJIS_COMBO_UNOFFICIAL = "<:unofficial:1233866785862848583><:unofficial_1:1233866787314073781><:unofficial_2:1233866788777754644>"

class DiscordBotResponse:
    def __init__(self, bot_data: KamiChan, verbose: bool=False):
        self.verbose = verbose
        self.bot_data = bot_data
        self.verbose_log = ""

    def log_verbose(self, text: str, *, category: str|None = None):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if category:
            self.verbose_log += f"[{current_time}] --- {category} ---\n{text}\n"
        else:
            self.verbose_log += f"[{current_time}] {text}\n"

    async def create_or_fallback(self, message: discord.Message, model_names: list[str]) -> str:
        all_errors = []
        full_prompt = await self.build_full_prompt(
            self.bot_data.memory.without_dupe_ending_user_msgs(), 
            message
        )
        
        for model_name in model_names:
            try:
                response = await self.bot_data.clients[MAIN_CLIENT_NAME].send_request(
                    LLMRequest(
                        prompt=Prompt(messages=full_prompt),
                        model_name=model_name,
                        max_tokens=2000,
                        temperature=0
                    )
                )
                response_txt = response.message.content
                self.log_verbose(response_txt, category="PERSONALITY-LESS MESSAGE")
                self.log_verbose(f"Length (chars): {len(response_txt)}")
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
        response = await self.bot_data.clients[PERSONALITY_REWRITER_NAME].send_request(
            LLMRequest(
                prompt=Prompt(messages=prompts.REWRITER_PROMPT.replace("<message>", message).to_openai_format()),
                model_name="meta-llama/llama-3.1-405b-instruct",
                max_tokens=2000,
                temperature=0.3
            )
        )  
        return response.message.content \
            .strip() \
            .removeprefix("REWRITTEN: ") \
            .replace("<+1>", random.choice(["<:paperUwU:1018366709658308688>", "<:Paperyis:1022991557978238976>", "<:Paperyis:1022991557978238976>"])) \
            .replace("<-1>", random.choice(["<a:notlikepaper:1165467302578360401>"])) \
            .replace("<0>",  random.choice(["<:paperOhhh:1018366673423695872>"]))

    async def fetch_last_user_query(self, model: LLMProvider) -> str:
        user_prompt_str: str = ""
        user_prompt_str = "\n".join(
            [memorized_message.text for memorized_message in self.bot_data.memory.get_memory()]
        )
        last_user = self.bot_data.memory.get_memory()[-1].nick

        response_choice = await model.send_request(
            LLMRequest(
                prompt=prompts.QUERY_SUMMARIZER_PROMPT \
                    .replace("((user_query))", user_prompt_str) \
                    .replace("((last_user))", last_user),
                model_name='meta-llama/llama-3.1-405b-instruct',
                temperature=0.2
            )
        )
        return response_choice.message.content

    async def summarize_relevant_facts(self, model: LLMProvider, user_query: str) -> str | None:
        user_prompt_str = ""
        sanitized_msg = await self.bot_data.sanitize_str(user_query)
        knowledge_list = await self.bot_data.vector_db.search(sanitized_msg, 5, "knowledge")

        if len(knowledge_list) == 0:
            return None

        for knowledge in knowledge_list:
            user_prompt_str += "INFO: \n" + str(knowledge) # TODO: what's the type of this?

        self.log_verbose(f"--- DATABASE CLOSEST MATCHES ---\n{user_prompt_str}\n")
        user_prompt_str += "QUERY: " + user_query
        response_choice = await model.send_request(
            LLMRequest(
                prompt=prompts.INFO_SELECTOR_PROMPT \
                    .replace("((user_query))", user_prompt_str),
                model_name='gpt-4o-mini'
            )
        )
        # queries_manager = await preset_queries.manager(model)
        known_query_info = ""
        # for preset in await queries_manager.get_all_matching_user_utterance(user_query):
        #    known_query_info += preset.answer + "\n"
        return f"{response_choice.message.content}\n{known_query_info}"

    async def describe_image_if_present(self, message) -> str | None:
        if len(message.attachments) == 1:
            if message.channel.nsfw:
                await message.reply(":X: I can't see attachments in NSFW channels!")
                return None
            attachment = message.attachments[0]
            if attachment.content_type.startswith("image/"):
                await message.add_reaction("👀")
                # Todo: only last message is possibly not enough context
                response = await self.bot_data.clients[IMAGE_VIEWER_NAME].send_request(
                    LLMRequest(
                        prompt=Prompt(
                                messages=[
                                    Prompt.user_msg(
                                        content=f"Describe the image in a sufficient way to answer the following query: '{message.content}'" \
                                        "If the query is empty, just describe the image. ",
                                        image_url=attachment.url
                                    )
                                ]
                            ),
                         model_name="google/gemini-pro-1.5"
                    )
                )
                return response.message.content

    async def build_full_prompt(self, memory_snapshot: BotMemory, original_msg: discord.Message) -> list[Any]:
        now_str = datetime.datetime.now().strftime("%B %d, %H:%M:%S")
        model = self.bot_data.clients[MAIN_CLIENT_NAME]

        # TODO: the "Prompt" class is weirdly used here, but not worth looking too much into as of this will be replaced by a JSON file eventually
        system_prompt_str = prompts.KAMI_CHAN_PROMPT \
            .replace("((nick))", memory_snapshot.get_memory()[-1].nick) \
            .replace("((now))", now_str).messages[0]["content"]

        user_query = await self.fetch_last_user_query(model)
        self.log_verbose(user_query, category="USER QUERY")

        knowledge = await self.summarize_relevant_facts(model, user_query)

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

        for memorized_message in memory_snapshot.get_memory():
            if memorized_message.is_bot:
                prompt.append(Prompt.assistant_msg(memorized_message.text))
            else:
                prompt.append(Prompt.user_msg(memorized_message.text))

        img_desc = await self.describe_image_if_present(original_msg)
        if img_desc:
            prompt.append(Prompt.user_msg(img_desc))

        self.log_verbose(f"--- FULL PROMPT ---\n{str(prompt)}\n")
        return prompt