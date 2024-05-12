from memorized_message import MemorizedMessage
from ai import OAICompatibleProvider
from ai_bot import AIBotData
from typing import Any
from datetime import datetime
from vector_db import QdrantVectorDbConnection
from . import prompts
import providers
import random, discord, openai

#
# TODO: This is a mess that has to be replaced with something data-driven
#
MAIN_CLIENT_NAME = "KAMI_CHAN_MAIN"
KNOWLEDGE_EXTRACTOR_NAME = "KAMI_CHAN_KNOWLEDGE_EXTRACTOR"
PERSONALITY_REWRITER_NAME = "KAMI_CHAN_PERSONALITY_REWRITER"
IMAGE_VIEWER_NAME = "KAMI_CHAN_IMAGE_VIEWER"

class KamiChan(AIBotData):
    def __init__(self,
                 name: str,
                 vector_db_conn: QdrantVectorDbConnection,
                 discord_bot_id: int):
        super().__init__(name, [])
        self.clients: dict[str, OAICompatibleProvider] = {}
        self._create_clients()
        self.discord_bot_id = discord_bot_id
        self.vector_db_conn = vector_db_conn
        self.memory = []
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
    
    async def memorize_short_term(self, message: discord.Message):
        self.memory.append(await MemorizedMessage.of_discord_message(message, self.sanitize_msg))
        if len(self.memory) > self.bot_data.RECENT_MEMORY_LENGTH:
            self.memory.pop(0)

    async def forget_short_term(self, message: discord.Message):
        self.memory = [mem_msg for mem_msg in self.memory if mem_msg.message_id != message.id ]  


class DiscordBotResponse:
    def __init__(self, bot_data: KamiChan):
        self.verbose = False
        self.bot_data = bot_data
        self.verbose_log = ""

    async def create(self, message: discord.Message) -> str:
        self.verbose = message.content.endswith("--v")
        full_prompt = await self.build_full_prompt(self.bot_data.memory, message)
        response = await self.bot_data.clients[MAIN_CLIENT_NAME].generate_response(
            prompt=full_prompt,
            model="openai/gpt-3.5-turbo-0125",
            max_tokens=300,
            temperature=0.2
        )
        response_txt = response.message.content
        personality_rewrite = await self.personality_rewrite(response_txt)
        self.log_verbose(f"--- IN-CHARACTER REWRITE ---\n{personality_rewrite}\n")
        return personality_rewrite

    async def log_verbose(self, text: str):
        self.verbose_log += text + "\n"

    async def personality_rewrite(self, message: str) -> str:
        response = await self.bot_data.clients[PERSONALITY_REWRITER_NAME].generate_response(
            prompt=prompts.REWRITER_PROMPT.replace("<message>", message).to_openai_format(),
            model="anthropic/claude-3-haiku",
            max_tokens=500,
            temperature=0.3,
            #logit_bias={
            #    "9712": -8,  # super
            #    "35734": 3,  # why
            #    "8823": -10, # help
            #    "8823": -10, # assist
            #    "2000": 3,   # for
            #    "1568": 3,   # try
            #    "15873": 7   # [[
            #}
        )
        content = response.message.content
        content = content \
            .replace("[[+1]]", random.choice(["<:paperUwU:1018366709658308688>", "<:Paperyis:1022991557978238976>", "<:Paperyis:1022991557978238976>"])) \
            .replace("[[-1]]", random.choice(["<a:notlikepaper:1165467302578360401>"])) \
            .replace("[[0]]",  random.choice(["<:paperOhhh:1018366673423695872>"]))

        return content

    async def fetch_last_user_query(self, model: OAICompatibleProvider) -> str:
        user_prompt_str: str = ""
        for memorized_message in self.bot_data.memory:
            user_prompt_str += f"\n{memorized_message.text}"
        last_user = self.bot_data.memory[-1].nick

        response_choice = await model.generate_response(
            prompt=prompts.QUERY_SUMMARIZER_PROMPT \
                .replace("((user_query))", user_prompt_str) \
                .replace("((last_user))", last_user).to_openai_format(),
            model='cohere/command-r'
        )
        return response_choice.message.content

    async def summarize_relevant_facts(self,  model: OAICompatibleProvider, user_query: str):
        user_prompt_str = ""
        knowledge_list = await self.bot_data.vector_db_conn.query_relevant_knowledge(await self.sanitize_str(user_query))
        for knowledge in knowledge_list:
            user_prompt_str += "INFO: \n" + knowledge.payload
        user_prompt_str += "QUERY: " + user_query
        response_choice = await model.generate_response(
            prompt=prompts.INFO_SELECTOR_PROMPT \
                .replace("((user_query))", user_prompt_str).to_openai_format(),
            model='gpt-3.5-turbo'
        )
        return response_choice.message.content

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

    async def build_full_prompt(self, memory_snapshot: list[MemorizedMessage], original_msg: discord.Message) -> list[Any]:
        now_str = datetime.now().strftime("%B %d, %H:%M:%S")
        model = self.bot_data.clients[MAIN_CLIENT_NAME]

        # TODO: the "Prompt" class is weirdly used here, but not worth looking too much into as of this will be replaced by a JSON file eventually
        system_prompt_str = prompts.KAMI_CHAN_PROMPT \
            .replace("((nick))", memory_snapshot[-1].nick) \
            .replace("((now))", now_str)._dict[0]["content"]

        user_query = await self.fetch_last_user_query(model)
        self.log_verbose(f"--- USER QUERY ---\n{user_query}\n")
        knowledge = await self.summarize_relevant_facts(model, user_query)
        system_prompt_str += f"\n[INFO FROM KNOWLEDGE DB]:\n{knowledge}\n"
        prompt: list[Any] = [
            OAICompatibleProvider.system_msg(system_prompt_str)
        ]
        self.log_verbose(f"--- INFO FROM KNOWLEDGE DB ---\n{knowledge}\n")
        old_messages_str = ""
        for old_message in await self.bot_data.vector_db_conn.query_relevant_messages(original_msg.content):
            old_messages_str += f"[{old_message.sent.isoformat()} by {old_message.nick}] {old_message.text}"

        system_prompt_str += \
        f"""
        [[OLD MESSAGES IN YOUR MEMORY]]:\n{old_messages_str}\n\n
        [[RECENT CONVERSATION HISTORY]]:\n{knowledge}
        """

        for memorized_message in memory_snapshot:
            if memorized_message.is_bot:
                prompt.append(OAICompatibleProvider.assistant_msg(memorized_message.text))
            else:
                prompt.append(OAICompatibleProvider.user_msg(memorized_message.text))

        img_desc = await self.describe_image_if_present(original_msg)
        if img_desc:
            prompt.append(OAICompatibleProvider.user_msg(img_desc))

        self.log_verbose(f"--- FULL PROMPT ---\n{str(prompt)}\n")
        return prompt