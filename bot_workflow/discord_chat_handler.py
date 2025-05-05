import io
import discord
import traceback

from typing import Tuple
from discord.ext import commands
from util.rate_limits import RateLimiter, RateLimit
from bot_workflow.response_logs import ResponseLogsManager
from bot_workflow.memorized_message import MemorizedMessage
from bot_workflow.ai_bot import CustomBotData, DiscordBotResponse
from bot_workflow.discord_message_parser import DiscordMessageParser, DenialReason, SpecialFunctionFlags

MSG_LOG_FILE_REPLY = "Verbose logs for message ID {} attached (only last 10 are stored)"
MSG_INVALID_LOG_REQUEST = ":x: Expected a message ID before --l, not '{}'"

class DiscordChatHandler(commands.Cog):
    def __init__(self, discord_bot: commands.Bot, ai_bot_data: CustomBotData):
        self.bot: commands.Bot = discord_bot
        self.rate_limiter = RateLimiter(
            RateLimit(n_messages=3, seconds=10),
            RateLimit(n_messages=10, seconds=60),
            RateLimit(n_messages=35, seconds=5 * 60),
            RateLimit(n_messages=100, seconds=2 * 3600),
            RateLimit(n_messages=250, seconds=8 * 3600)
        )
        self.logs = ResponseLogsManager()
        self.message_parser = DiscordMessageParser(self.bot)
        self.ai_bot = ai_bot_data

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot: 
            return
        
        ctx = self.message_parser.parse_message(message)
        if ctx.denial_reason == DenialReason.DID_NOT_PING:
            return
        if ctx.denial_reason == DenialReason.RATE_LIMITED:
            await message.reply("You are rate limited, please wait")
            return
        if ctx.denial_reason == DenialReason.TOO_LONG:
            for emoji in ['🇹', '🇱', '🇩', '🇷']:
                await message.add_reaction(emoji)
            return
        if SpecialFunctionFlags.VIEW_MESSAGE_LOGS in ctx.called_functions:
            await self.handle_log_request(message)
            return
        
        verbose = SpecialFunctionFlags.REQUEST_VERBOSE_REPLY in ctx.called_functions
        await self.respond_with_llm(message, verbose=verbose)

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

    async def memorize_message(self, message: MemorizedMessage, *, pending: bool, add_after_id: None | int) -> None:
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
        self.ai_bot.long_term_memory.memorize(message)
        
    async def memorize_discord_message(self, message: discord.Message, *, pending: bool, add_after_id: None | int) -> None:
        to_memorize = await MemorizedMessage.of_discord_message(message)
        await self.memorize_message(
            to_memorize,
            pending=pending,
            add_after_id=add_after_id
        )
        self.ai_bot.long_term_memory.memorize(to_memorize)

    async def handle_error(self, message: discord.Message, reply: discord.Message, error: Exception):
        # TODO: implement message forgetting
        # await self.forget_message(message)
        # await self.forget_message(reply)
        traceback.print_exc()
        await reply.edit(content=f"There was an error: ```{str(error)}```") # TODO: send lang message if possible