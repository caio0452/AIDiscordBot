import io
import re
import discord
import traceback

from io import StringIO
from discord.ext import commands
from core.util.rate_limits import RateLimiter, RateLimit
from core.bot_workflow.message_snapshot import MessageSnapshot
from core.bot_workflow.ai_bot import CustomBotData, AIDiscordBotResponder
from core.bot_workflow.response_logs import ResponseLogsManager, SimpleDebugLogger
from core.bot_workflow.discord_message_parser import DiscordMessageParser, DenialReason, SpecialFunctionFlags

MSG_LOG_FILE_REPLY = "Verbose logs for message ID {} attached (only last 10 are stored)"

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
        self.message_parser = DiscordMessageParser(self.bot)
        self.ai_bot = ai_bot_data
        self.logger = SimpleDebugLogger("ChatHandlerLogger")

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
        ctx = self.message_parser.parse_message(message)
        try:
            num = None
            for num_str in ctx.sanitized_content.split(" "):
                if num_str.isdigit():
                    num = int(num_str)
                    break

            if num is None:
                await message.reply("❌ No numerical message ID found")
                return
            
            log_data = ResponseLogsManager.instance().get_log_by_id(num)
            if log_data is None:
                await message.reply(f"❌ No log with ID {num} found")
                return
            log_file = io.BytesIO(log_data.encode('utf-8'))
            await message.reply(
                content=MSG_LOG_FILE_REPLY.format(num),
                files=[discord.File(log_file, filename="verbose_log.txt")]
            )
        except ValueError:
            invalid_log_msg = self.ai_bot.profile.lang["invalid_log_request"]
            await message.reply(invalid_log_msg.format(sanitized_msg))

    async def respond_with_llm(self, message: discord.Message, *, verbose: bool=False):
        await self.memorize_discord_message(message, pending=True, add_after_id=None)

        reply = await message.reply(
            self.ai_bot.profile.lang["bot_typing"], 
            silent=self.ai_bot.profile.options.only_ping_on_response_finish
        )
        
        try:
            resp = await self.generate_response(message, verbose)
            resp_msg: discord.Message = await self.reply_chunked_with_disclaimers(reply, resp.text, ping=True)

            if verbose:
                log_file = StringIO(resp.verbose_log_output)
                await resp_msg.edit(attachments=[discord.File(log_file, filename="log.txt")])
  
            await self.memorize_message(
                MessageSnapshot(
                    text=resp.text,  
                    nick=resp_msg.author.name,
                    sent=resp_msg.created_at,
                    is_bot=True,
                    message_id=resp_msg.id 
                ),
                pending=False,
                add_after_id=message.id
            )
            await self.ai_bot.recent_history.mark_finalized(message.id)
            ResponseLogsManager.instance().store_log(reply.id, resp.verbose_log_output)
        except Exception as e:
            await self.handle_error(message, reply, e)

    async def generate_response(self, message: discord.Message, verbose: bool) -> AIDiscordBotResponder.Response:
        resp = AIDiscordBotResponder(self.ai_bot, message, verbose)
        return await resp.create_response()

    def _chunk_by_length_and_spaces(self, full_text: str, max_chunk_length: int) -> list[str]:
        chunks: list[str] = []
        cursor = 0
        text_length = len(full_text)

        while cursor < text_length:
            segment_end = min(cursor + max_chunk_length, text_length)
            segment = full_text[cursor:segment_end]

            if segment_end < text_length:
                last_space = segment.rfind(' ')
                # If there are no spaces in the last half, maybe the string isn't meant to be split on spaces
                min_space_pos_to_split = max_chunk_length // 2
                if last_space > min_space_pos_to_split:
                    end = cursor + last_space
                    segment = full_text[cursor:end]

            chunks.append(segment)
            cursor = segment_end

        return chunks
    
    def _balance_code_block_fences(self, *, original_text: str, chunked_text: list[str]) -> list[str]:
        fence_pattern = re.compile(r"```(\w+)?")
        balanced: list[str] = []
        inside_code = False

        def last_fence_language(text: str, *, before_pos: int) -> str:
            matches = list(fence_pattern.finditer(text, 0, before_pos))
            if not matches:
                return ""
            lang = matches[-1].group(1)
            return lang or ""

        cursor = 0
        for chunk in chunked_text:
            fence_count = len(fence_pattern.findall(chunk))
            
            if fence_count % 2 == 1:
                inside_code = not inside_code
                chunk += "\n```"

            # If inside a code block after closing, next chunk must reopen
            if inside_code:
                lang = last_fence_language(original_text, before_pos=cursor)
                chunk = f"```{lang}\n" + chunk

            balanced.append(chunk)
            cursor += len(chunk)

        return balanced

    async def reply_chunked_with_disclaimers(self, reply: discord.Message, resp_str: str, *, ping: bool) -> discord.Message:
        disclaimer = self.ai_bot.profile.lang.get("disclaimer", "")
        max_chunk_length = 1800 - len(disclaimer)

        def strip_newline(chunk):
            return chunk.strip('\r\n') if self.ai_bot.profile.options.remove_trailing_newline else chunk

        raw_chunks = self._chunk_by_length_and_spaces(resp_str, max_chunk_length)
        code_balanced_chunks = [
            f"{strip_newline(chunk)}{disclaimer}" 
            for chunk in self._balance_code_block_fences(original_text=resp_str, chunked_text=raw_chunks)
        ]

        if self.ai_bot.profile.options.only_ping_on_response_finish:
            last_msg = await reply.reply(content=code_balanced_chunks[0], silent=False)
            await reply.delete()
        else:
            last_msg = await reply.edit(content=code_balanced_chunks[0])

        for chunk in code_balanced_chunks[1:]:
            last_msg = await reply.reply(content=chunk, silent=not ping)
        return last_msg
    
    async def memorize_message(self, message: MessageSnapshot, *, pending: bool, add_after_id: None | int) -> None:
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
        if self.ai_bot.long_term_memory is not None:
            await self.ai_bot.long_term_memory.memorize(message)
        
    async def memorize_discord_message(self, message: discord.Message, *, pending: bool, add_after_id: None | int) -> None:
        to_memorize = await MessageSnapshot.of_discord_message(message)
        await self.memorize_message(
            to_memorize,
            pending=pending,
            add_after_id=add_after_id
        )
        if self.ai_bot.long_term_memory is not None:
            await self.ai_bot.long_term_memory.memorize(to_memorize)

    async def handle_error(self, message: discord.Message, reply: discord.Message, error: Exception):
        # TODO: implement message forgetting
        # await self.forget_message(message)
        # await self.forget_message(reply)
        traceback.print_exc()
        await reply.edit(content=f"There was an error: ```{str(error)[:1000]}```") # TODO: send lang message if possible