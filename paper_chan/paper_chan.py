from memorized_message import MemorizedMessage
from ai import OAICompatibleProvider, OpenAIModerator, GPT4Vision
from ai_bot import AIBot
from typing import Any
from datetime import datetime
from vector_db import QdrantVectorDbConnection
from . import prompts
import random, discord, openai

#
# TODO: This is a mess that has to be replaced with something data-driven
#
class PaperChan(AIBot):
    def __init__(self,
                 name: str,
                 openai_client: openai.AsyncOpenAI,
                 vector_db_conn: QdrantVectorDbConnection,
                 discord_bot_id: int):
        super().__init__(name, [])
        self.discord_bot_id = discord_bot_id
        self.vector_db_conn = vector_db_conn
        self.memory = []
        self.RECENT_MEMORY_LENGTH = 5
        self.moderator = OpenAIModerator(openai_client)
        self.image_viewer = GPT4Vision(openai_client)
        self.client = OAICompatibleProvider(openai_client)

    async def respond_to_query(self, message: discord.Message) -> str:
        flagged = await self.moderator.is_flagged(message.content)
        if flagged:
            return "<:paperdisgusts:1165462194046111774> `Message not processed: flagged by moderation API`"
        full_prompt = await self.build_full_prompt(self.memory, self.client, message)
        response = await self.client.generate_response(
            prompt=full_prompt,
            model="gpt-3.5-turbo-0125",
            max_tokens=300,
            temperature=0.5
        )
        response_txt = response.message.content
        personality_rewrite = await self.personality_rewrite(response_txt)
        return personality_rewrite

    async def personality_rewrite(self, message: str) -> str:
        response = await self.client.generate_response(
            prompt=prompts.REWRITER_PROMPT.replace("<message>", message).to_openai_format(),
            model="gpt-3.5-turbo-0125",
            max_tokens=300,
            temperature=0.4,
            logit_bias={
                "9712": -8,  # super
                "35734": 3,  # why
                "8823": -10, # help
                "8823": -10, # assist
                "2000": 3,   # for
                "1568": 3,   # try
                "15873": 7   # [[
            }
        )
        content = response.message.content
        content = content \
            .replace("[[+1]]", random.choice(["<:paperUwU:1018366709658308688>", "<:Paperyis:1022991557978238976>", "<:Paperyis:1022991557978238976>"])) \
            .replace("[[-1]]", random.choice(["<a:notlikepaper:1165467302578360401>"])) \
            .replace("[[0]]",  random.choice(["<:paperOhhh:1018366673423695872>"]))

        return content

    async def memorize_short_term(self, message: discord.Message):
        self.memory.append(await MemorizedMessage.of_discord_message(message, self.sanitize_msg))
        if len(self.memory) > self.RECENT_MEMORY_LENGTH:
            self.memory.pop(0)

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

    async def fetch_last_user_query(self, model: OAICompatibleProvider) -> str:
        user_prompt_str: str = ""
        for memorized_message in self.memory:
            user_prompt_str += f"\n{memorized_message.text}"
        last_user = self.memory[-1].nick

        response_choice = await model.generate_response(
            prompt=prompts.QUERY_SUMMARIZER_PROMPT \
                .replace("((user_query))", user_prompt_str) \
                .replace("((last_user))", last_user).to_openai_format(),
            model='gpt-3.5-turbo-0125'
        )
        return response_choice.message.content

    async def summarize_relevant_facts(self,  model: OAICompatibleProvider, user_query: str):
        user_prompt_str = ""
        knowledge_list = await self.vector_db_conn.query_relevant_knowledge(await self.sanitize_str(user_query))
        for knowledge in knowledge_list:
            user_prompt_str += "INFO: \n" + knowledge.payload
        user_prompt_str += "QUERY: " + user_query
        response_choice = await model.generate_response(
            prompt=prompts.INFO_SELECTOR_PROMPT \
                .replace("((user_query))", user_prompt_str).to_openai_format(),
            model='gpt-3.5-turbo-0125'
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
                response = await self.image_viewer.describe_image(attachment.url, message.content)
                return response.message.content

    async def build_full_prompt(self, memory_snapshot: list[MemorizedMessage], model: OAICompatibleProvider, original_msg: discord.Message) -> list[Any]:
        now_str = datetime.now().strftime("%B %d, %H:%M:%S")

        # TODO: the "Prompt" class is weirdly used here, but not worth looking too much into as of this will be replaced by a JSON file eventually
        system_prompt_str = prompts.PAPER_CHAN_PROMPT \
            .replace("((nick))", memory_snapshot[-1].nick) \
            .replace("((now))", now_str)._dict[0]["content"]

        user_query = await self.fetch_last_user_query(model)
        knowledge = await self.summarize_relevant_facts(model, user_query)
        system_prompt_str += f"\n[INFO FROM KNOWLEDGE DB]:\n{knowledge}\n"
        prompt: list[Any] = [
            OAICompatibleProvider.system_msg(system_prompt_str)
        ]
        old_messages_str = ""
        for old_message in await self.vector_db_conn.query_relevant_messages(original_msg.content):
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

        return prompt