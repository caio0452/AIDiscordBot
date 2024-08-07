from memorized_message import MemorizedMessage
from ai import OAICompatibleProvider
from ai_bot import AIBotData, BotMemory
from typing import Any
from datetime import datetime
from vector_db import QdrantVectorDbConnection
from . import prompts
import providers
import random, discord, openai
import preset_queries
import datetime

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
                 vector_db_conn: QdrantVectorDbConnection,
                 discord_bot_id: int):
        super().__init__(name, BotMemory())
        self.clients: dict[str, OAICompatibleProvider] = {}
        self._create_clients()
        self.discord_bot_id = discord_bot_id
        self.vector_db_conn = vector_db_conn
        self.memory: BotMemory = BotMemory()
        self.RECENT_MEMORY_LENGTH = 5
        
    def _create_clients(self):
        required_provider_names = [
           MAIN_CLIENT_NAME, KNOWLEDGE_EXTRACTOR_NAME, PERSONALITY_REWRITER_NAME, IMAGE_VIEWER_NAME
        ]
        for name in required_provider_names:
            provider = providers.get_provider_by_name(name)
            client = OAICompatibleProvider(openai.AsyncOpenAI(
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
        full_prompt = await self.build_full_prompt(
            self.bot_data.memory.without_dupe_ending_user_msgs(), 
            message
        )
        
        for model_name in model_names:
            try:
                response = await self.bot_data.clients[MAIN_CLIENT_NAME].generate_response(
                    prompt=full_prompt,
                    model=model_name,
                    max_tokens=2000,
                    temperature=0
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
    
        raise RuntimeError("Could not generate response")

    async def create(self, message: discord.Message) -> str:
        return await self.create_or_fallback(message, ["google/gemini-pro-1.5-exp", "openai/gpt-4o-mini", "meta-llama/llama-3.1-405b-instruct"])

    async def personality_rewrite(self, message: str) -> str:
        response = await self.bot_data.clients[PERSONALITY_REWRITER_NAME].generate_response(
            prompt=prompts.REWRITER_PROMPT.replace("<message>", message).to_openai_format(),
            model="meta-llama/llama-3.1-405b-instruct",
            max_tokens=2000,
            temperature=0.3,
        )  
        return response.message.content \
            .strip() \
            .removeprefix("REWRITTEN: ") \
            .replace("<+1>", random.choice(["<:paperUwU:1018366709658308688>", "<:Paperyis:1022991557978238976>", "<:Paperyis:1022991557978238976>"])) \
            .replace("<-1>", random.choice(["<a:notlikepaper:1165467302578360401>"])) \
            .replace("<0>",  random.choice(["<:paperOhhh:1018366673423695872>"]))

    async def fetch_last_user_query(self, model: OAICompatibleProvider) -> str:
        user_prompt_str: str = ""
        user_prompt_str = "\n".join(
            [memorized_message.text for memorized_message in self.bot_data.memory.get_memory()]
        )
        last_user = self.bot_data.memory.get_memory()[-1].nick

        response_choice = await model.generate_response(
            prompt=prompts.QUERY_SUMMARIZER_PROMPT \
                .replace("((user_query))", user_prompt_str) \
                .replace("((last_user))", last_user).to_openai_format(),
            model='meta-llama/llama-3.1-405b-instruct',
            temperature=0.2
        )
        return response_choice.message.content

    async def summarize_relevant_facts(self, model: OAICompatibleProvider, user_query: str) -> str | None:
        user_prompt_str = ""
        knowledge_list = await self.bot_data.vector_db_conn.query_relevant_knowledge(
            await self.bot_data.sanitize_str(user_query))
        if len(knowledge_list) == 0:
            return None
        for knowledge in knowledge_list:
            user_prompt_str += "INFO: \n" + knowledge.payload
        self.log_verbose(f"--- DATABASE CLOSEST MATCHES ---\n{user_prompt_str}\n")
        user_prompt_str += "QUERY: " + user_query
        response_choice = await model.generate_response(
            prompt=prompts.INFO_SELECTOR_PROMPT \
                .replace("((user_query))", user_prompt_str).to_openai_format(),
            model='gpt-4o-mini'
        )
        queries_manager = await preset_queries.manager(model)
        known_query_info = ""
        for preset in await queries_manager.get_all_matching_user_utterance(user_query):
            known_query_info += preset.answer + "\n"
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
                response = await self.bot_data.clients[IMAGE_VIEWER_NAME].describe_image(attachment.url, message.content)
                return response.message.content

    async def build_full_prompt(self, memory_snapshot: BotMemory, original_msg: discord.Message) -> list[Any]:
        now_str = datetime.datetime.now().strftime("%B %d, %H:%M:%S")
        model = self.bot_data.clients[MAIN_CLIENT_NAME]

        # TODO: the "Prompt" class is weirdly used here, but not worth looking too much into as of this will be replaced by a JSON file eventually
        system_prompt_str = prompts.KAMI_CHAN_PROMPT \
            .replace("((nick))", memory_snapshot.get_memory()[-1].nick) \
            .replace("((now))", now_str)._dict[0]["content"]

        user_query = await self.fetch_last_user_query(model)
        self.log_verbose(user_query, category="USER QUERY")

        knowledge = await self.summarize_relevant_facts(model, user_query)

        if knowledge is not None:
            system_prompt_str += f"\n[INFO FROM KNOWLEDGE DB]:\n{knowledge}\n"
            self.log_verbose(knowledge, category="INFO FROM KNOWLEDGE DB")
        else:
            self.log_verbose("The knowledge database has nothing relevant", category="INFO FROM KNOWLEDGE DB")

        prompt: list[Any] = [
            OAICompatibleProvider.system_msg(system_prompt_str)
        ]
        old_messages_str = ""
        for old_message in await self.bot_data.vector_db_conn.query_relevant_messages(original_msg.content):
            old_messages_str += f"[{old_message.sent.isoformat()} by {old_message.nick}] {old_message.text}"

        system_prompt_str += \
        f"""
        [[OLD MESSAGES IN YOUR MEMORY]]:\n{old_messages_str}\n\n
        [[RECENT CONVERSATION HISTORY]]:\n{knowledge}
        """

        for memorized_message in memory_snapshot.get_memory():
            if memorized_message.is_bot:
                prompt.append(OAICompatibleProvider.assistant_msg(memorized_message.text))
            else:
                prompt.append(OAICompatibleProvider.user_msg(memorized_message.text))

        img_desc = await self.describe_image_if_present(original_msg)
        if img_desc:
            prompt.append(OAICompatibleProvider.user_msg(img_desc))

        self.log_verbose(f"--- FULL PROMPT ---\n{str(prompt)}\n")
        return prompt