import io
import discord
import traceback

from typing import Tuple
from discord.ext import commands
from providers import ProviderStore
from vector_db import VectorDatabase
from rate_limits import RateLimit, RateLimiter
from memorized_message import MemorizedMessage
from kami_chan.kami_chan import KamiChan, DiscordBotResponse

BOT_NAME = "Kami-Chan"
MAX_CHAT_CHARACTERS = 1000
MSG_RATE_LIMITED = f"{KamiChan.Vocabulary.EMOJI_NO} :x: `You are being rate limited`"
MSG_BOT_TYPING = f"-# {KamiChan.Vocabulary.EMOJI_UWU} {BOT_NAME} is typing..."
MSG_ERROR = f"Sorry, there was an error!! {KamiChan.Vocabulary.EMOJI_DESPAIR} ```{{}}```"
MSG_DISCLAIMER = f"-# Unofficial bot. FICTITIOUS AI-generated content. | [Learn more.](https://discord.com/channels/532557135167619093/1192649325709381673/1196285641978302544)"
MSG_LOG_FILE_REPLY = "Verbose logs for message ID {} attached (only last 10 are stored)"
MSG_INVALID_LOG_REQUEST = ":x: Expected a message ID before --l, not '{}'"
MODEL_REQUEST_ORDER = ["google/gemini-flash-1.5", "qwen/qwen-2-72b-instruct", "meta-llama/llama-3-70b-instruct"]

class MessageFlag:
    BOT_MESSAGE = "BOT_MESSAGE"
    LOG_REQUEST = "LOG_REQUEST"
    VERBOSE_REQUEST = "VERBOSE_REQUEST"
    TOO_LONG = "TOO_LONG"
    RATE_LIMITED = "RATE_LIMITED"
    PINGED_BOT = "PINGED_BOT"

class ChatHandler(commands.Cog):
    def __init__(self, bot: commands.Bot, provider_store: ProviderStore, vector_database: VectorDatabase):
        self.bot: commands.Bot = bot
        self.RECENT_MEMORY_LENGTH = 5
        self.rate_limiter = RateLimiter(
            RateLimit(n_messages=3, seconds=10),
            RateLimit(n_messages=10, seconds=60),
            RateLimit(n_messages=35, seconds=5 * 60),
            RateLimit(n_messages=100, seconds=2 * 3600),
            RateLimit(n_messages=250, seconds=8 * 3600)
        )
        self.ai_bot = KamiChan(BOT_NAME, vector_database, provider_store, bot.user.id)
        self._last_message_id_logs: dict[int, str] = {}
        self.vector_database = vector_database

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
        message_flags = self.get_message_flags(message)
        
        if MessageFlag.BOT_MESSAGE in message_flags: 
            return
        if not MessageFlag.PINGED_BOT in message_flags:
            return
        elif MessageFlag.RATE_LIMITED in message_flags: 
            return
        elif MessageFlag.LOG_REQUEST in message_flags:
            await self.handle_log_request(message)
        elif MessageFlag.TOO_LONG in message_flags:
            await self.handle_too_long_message(message)
        elif MessageFlag.PINGED_BOT in message_flags:
            is_verbose = (MessageFlag.VERBOSE_REQUEST in message_flags)
            await self.respond_with_llm(message, verbose=is_verbose)

    def cache_log(self, message_id: int, log: str):
        print(f"Saved log for message id {message_id}")
        self._last_message_id_logs[message_id] = log
        if len(self._last_message_id_logs) > 10:
            oldest_key = next(iter(self._last_message_id_logs))
            del self._last_message_id_logs[oldest_key]

    def get_log_by_id(self, message_id: int) -> str:
        return self._last_message_id_logs.get(message_id, "(NONE FOUND)")

    async def handle_log_request(self, message: discord.Message):
        sanitized_msg = message.content.strip().replace("--l", "")
        try:
            message_id = int(sanitized_msg)
            log_file = io.BytesIO(self.get_log_by_id(message_id).encode('utf-8'))
            await message.reply(
                content=MSG_LOG_FILE_REPLY.format(message_id),
                files=[discord.File(log_file, filename="verbose_log.txt")]
            )
        except ValueError:
            await message.reply(MSG_INVALID_LOG_REQUEST.format(sanitized_msg))

    async def handle_too_long_message(self, message: discord.Message):
        emojis = ['🇹', '🇱', '🇩', '🇷']
        for emoji in emojis:
            await message.add_reaction(emoji)

    async def respond_with_llm(self, message: discord.Message, *, verbose: bool=False):
        self.rate_limiter.register_request(message.author.id)
        await self.memorize_message(message)
        reply = await message.reply(MSG_BOT_TYPING)
        
        try:
            resp_str, verbose_log = await self.generate_response(message, verbose)
            if verbose:
                reply = await self.attach_log(reply, resp_str, verbose_log)
            await self.memorize_message(reply)
            await self.add_disclaimer(reply, resp_str)
            self.cache_log(reply.id, verbose_log)
        except Exception as e:
            await self.handle_error(message, reply, e)

    async def generate_response(self, message: discord.Message, verbose: bool) -> Tuple[str, str]:
        resp = DiscordBotResponse(self.ai_bot, verbose)
        resp_str = await resp.create_or_fallback(message, MODEL_REQUEST_ORDER)
        return resp_str, resp.verbose_log

    async def attach_log(self, reply: discord.Message, resp_str: str, verbose_log: str) -> discord.Message:
        if verbose_log:
            log_file = io.BytesIO(verbose_log.encode('utf-8'))
            return await reply.edit(
                content=resp_str, 
                attachments=[discord.File(log_file, filename="verbose_log.txt")]
            )
        else:
            return await reply.edit(content=resp_str)

    async def add_disclaimer(self, reply: discord.Message, resp_str: str):
        await reply.edit(content=f"{resp_str}\n{MSG_DISCLAIMER}")

    async def memorize_message(self, message: discord.Message):
        await self.ai_bot.memory.memorize_short_term(
            await MemorizedMessage.of_discord_message(message), 
            None
        )
        await self.vector_database.index(
            index_name="messages", 
            data=message.content, 
            metadata="", 
            entry_id=message.id
        )

    async def handle_error(self, message: discord.Message, reply: discord.Message, error: Exception):
        await self.memorize_message(message)
        await self.memorize_message(reply)
        traceback.print_exc()
        await reply.edit(content=MSG_ERROR.format(str(error)))