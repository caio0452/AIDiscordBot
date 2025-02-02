import io
import openai
import discord
import datetime 
import traceback

from typing import Tuple
from discord.ext import commands
from bot_workflow.presets.eta_classifier import EtaClassifier
from util.rate_limits import RateLimiter, RateLimit
from bot_workflow.memorized_message import MemorizedMessage
from bot_workflow.ai_bot import CustomBotData, DiscordBotResponse

BOT_NAME = "Kami-Chan"
MAX_CHAT_CHARACTERS = 1000
MSG_LOG_FILE_REPLY = "Verbose logs for message ID {} attached (only last 10 are stored)"
MSG_INVALID_LOG_REQUEST = ":x: Expected a message ID before --l, not '{}'"

class MessageFlag:
    BOT_MESSAGE = "BOT_MESSAGE"
    LOG_REQUEST = "LOG_REQUEST"
    VERBOSE_REQUEST = "VERBOSE_REQUEST" 
    TOO_LONG = "TOO_LONG"
    RATE_LIMITED = "RATE_LIMITED"
    PINGED_BOT = "PINGED_BOT"

class ResponseLogsManager:
    def __init__(self, log_capacity: int = 10):
        self.log_capacity = log_capacity
        self._last_message_id_logs: dict[int, str] = {}

    def store_log(self, message_id: int, log: str):
        print(f"Saved log for message id {message_id}")
        self._last_message_id_logs[message_id] = log
        if len(self._last_message_id_logs) > self.log_capacity:
            oldest_key = next(iter(self._last_message_id_logs))
            del self._last_message_id_logs[oldest_key]

    def get_log_by_id(self, message_id: int) -> str | None:
        return self._last_message_id_logs.get(message_id, None)

class DiscordChatHandler(commands.Cog):
    def __init__(self, discord_bot: commands.Bot, ai_bot_data: CustomBotData):
        self.bot: commands.Bot = discord_bot
        self.RECENT_MEMORY_LENGTH = 5
        self.rate_limiter = RateLimiter(
            RateLimit(n_messages=3, seconds=10),
            RateLimit(n_messages=10, seconds=60),
            RateLimit(n_messages=35, seconds=5 * 60),
            RateLimit(n_messages=100, seconds=2 * 3600),
            RateLimit(n_messages=250, seconds=8 * 3600)
        )
        self.logs = ResponseLogsManager()
        self.ai_bot = ai_bot_data
        self._last_message_id_logs: dict[int, str] = {}

    def get_message_flags(self, message: discord.Message) -> list[MessageFlag]:
        flags = [] 
        if message.author.bot:
            flags.append(MessageFlag.BOT_MESSAGE)

        if message.content.endswith("--l"):
            flags.append(MessageFlag.LOG_REQUEST)

        if message.content.endswith("--v"):
            flags.append(MessageFlag.VERBOSE_REQUEST)
        
        if len(message.content) > MAX_CHAT_CHARACTERS:
            flags.append(MessageFlag.TOO_LONG)
        
        if self.rate_limiter.is_rate_limited(message.author.id):
            flags.append(MessageFlag.RATE_LIMITED)
        
        if self.bot.user in message.mentions:
            flags.append(MessageFlag.PINGED_BOT)

        return flags

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        AUTORESPONDER_CHANNEL_ID = 1335411168948256852

        message_flags = self.get_message_flags(message)
        
        if message.channel.id == AUTORESPONDER_CHANNEL_ID:
            await self.run_autoresponder(message)
            return

        if MessageFlag.BOT_MESSAGE in message_flags: 
            return
        elif MessageFlag.RATE_LIMITED in message_flags: 
            return
        elif MessageFlag.LOG_REQUEST in message_flags:
            await self.handle_log_request(message)
        if MessageFlag.PINGED_BOT not in message_flags: # Handle logs even when bot is not pinged
            return
        elif MessageFlag.TOO_LONG in message_flags:
            await self.handle_too_long_message(message)
        elif MessageFlag.PINGED_BOT in message_flags:
            is_verbose = (MessageFlag.VERBOSE_REQUEST in message_flags)
            await self.respond_with_llm(message, verbose=is_verbose)

    async def handle_too_long_message(self, message: discord.Message):
        emojis = ['🇹', '🇱', '🇩', '🇷']
        for emoji in emojis:
            await message.add_reaction(emoji)

    async def handle_log_request(self, message: discord.Message):
        sanitized_msg = message.content.strip().replace("--l", "")
        try:
            message_id = int(sanitized_msg)
            log_data = self.logs.get_log_by_id(message_id)
            if log_data is None:
                await message.reply("No log with that ID found")
                return
            log_file = io.BytesIO(log_data.encode('utf-8'))
            await message.reply(
                content=MSG_LOG_FILE_REPLY.format(message_id),
                files=[discord.File(log_file, filename="verbose_log.txt")]
            )
        except ValueError:
            await message.reply(MSG_INVALID_LOG_REQUEST.format(sanitized_msg))

    async def respond_with_llm(self, message: discord.Message, *, verbose: bool=False):
        self.rate_limiter.register_request(message.author.id)
        await self.memorize_discord_message(message, pending=True, add_after_id=None)
        reply = await message.reply(self.ai_bot.profile.lang["bot_typing"])
        
        try:
            resp_str, verbose_log = await self.generate_response(message, verbose)
            # TODO: superfluous edits
            if verbose:
                reply = await self.attach_log(reply, resp_str, verbose_log)
            resp_msg: discord.Message = await self.send_discord_response(reply, resp_str)
            await self.memorize_message(
                MemorizedMessage(
                    text=resp_str,  
                    nick=resp_msg.author.name,
                    sent=resp_msg.created_at,
                    is_bot=True,
                    message_id=resp_msg.id 
                ),
                pending=False,
                add_after_id=message.id
            )
            await self.ai_bot.recent_history.mark_finalized(message.id)
            self.logs.store_log(reply.id, verbose_log)
        except Exception as e:
            await self.handle_error(message, reply, e)

    async def generate_response(self, message: discord.Message, verbose: bool) -> Tuple[str, str]:
        resp = DiscordBotResponse(self.ai_bot, verbose)
        resp_str = await resp.create(message)
        return resp_str, resp.logger.text

    async def attach_log(self, reply: discord.Message, resp_str: str, verbose_log: str) -> discord.Message:
        if verbose_log:
            log_file = io.BytesIO(verbose_log.encode('utf-8'))
            return await reply.edit(
                content=resp_str, 
                attachments=[discord.File(log_file, filename="verbose_log.txt")]
            )
        else:
            return await reply.edit(content=resp_str)

    async def send_discord_response(self, reply: discord.Message, resp_str: str) -> discord.Message:
        CHUNK_SIZE = 1800 
        chunks = []
        for i in range(0, len(resp_str), CHUNK_SIZE):
            chunks.append(resp_str[i:i + CHUNK_SIZE])

        if len(chunks) == 0:
            raise RuntimeError(f"Ended up with 0 chunks while trying to chunk message with content '{resp_str}'")
        
        last_message = chunks[0]
        disclaimer = self.ai_bot.profile.lang["disclaimer"]
        previous_message = await reply.edit(content=f"{last_message}{disclaimer}")  
        if len(chunks) >= 2:
            for chunk in chunks[1:]:
                previous_message = await reply.reply(content=chunk)

        return previous_message

    async def run_autoresponder(self, message: discord.Message):
        autoresponder_provider = self.ai_bot.provider_store.get_provider_by_name("DEFAULT")
        autoresponder_client = openai.AsyncOpenAI(
            api_key=autoresponder_provider.api_key, 
            base_url=autoresponder_provider.api_base, 
            timeout=60
        )
        classifier = await EtaClassifier.with_openai(     
            model="gpt-4o-mini",
            client=autoresponder_client
        )
        result = await classifier.classify(message.content)
        if result.belongs_to_class:
            await message.reply("Paper releases do not have any sort of ETA.")
        else:
            await message.reply("Sorry, I don't know how to respond to that yet")
        return
    
    async def memorize_message(self, message: MemorizedMessage, *, pending: bool, add_after_id: None | int):
        if add_after_id is None:
            await self.ai_bot.recent_history.add(
                message,
                pending=pending
            )
        else:
             await self.ai_bot.recent_history.add_after(
                add_after_id,
                message,
                pending=pending
            )
        # TODO: memorize long term

    async def memorize_discord_message(self, message: discord.Message, *, pending: bool, add_after_id: None | int):
        await self.memorize_message(
            await MemorizedMessage.of_discord_message(message),
            pending=pending,
            add_after_id=add_after_id
        )
        # TODO: memorize long term

    async def memorize_raw_message(self, *, text: str, nick: str, sent: datetime.datetime, is_bot: bool, message_id: int):
        await self.ai_bot.recent_history.add(
            MemorizedMessage(
                text=text,
                nick=nick,
                sent=sent,
                is_bot=is_bot,
                message_id=message_id
            )
        )
        # TODO: memorize long term

    async def handle_error(self, message: discord.Message, reply: discord.Message, error: Exception):
        # TODO: implement message forgetting
        # await self.forget_message(message)
        # await self.forget_message(reply)
        traceback.print_exc()
        await reply.edit(content=f"There was an error: ```{str(error)}```") # TODO: send custom message if possible