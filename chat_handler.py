import discord
import io
import traceback
import preset_queries

from discord.ext import commands
from vector_db import QdrantVectorDbConnection
from rate_limits import RateLimit, RateLimiter
from kami_chan.kami_chan import KamiChan, DiscordBotResponse

BOT_NAME = "Kami-Chan"

class ChatHandler(commands.Cog):
    MAX_CHAT_CHARACTERS = 500

    def __init__(self, bot: commands.Bot, db_connection: QdrantVectorDbConnection):
        self.bot: commands.Bot = bot
        self.vector_db_conn: QdrantVectorDbConnection = db_connection
        self.RECENT_MEMORY_LENGTH = 5
        self.rate_limiter = RateLimiter(
            RateLimit(n_messages=3, seconds=10),
            RateLimit(n_messages=10, seconds=60),
            RateLimit(n_messages=35, seconds=5 * 60),
            RateLimit(n_messages=100, seconds=2 * 3600),
            RateLimit(n_messages=250, seconds=8 * 3600)
        )
        self.ai_bot = KamiChan(BOT_NAME, db_connection, self.bot.user.id)
        self._last_message_id_logs: dict[int, str] = {}

    def cache_log(self, message_id: int, log: str):
        print(f"Saved log for message id {message_id}")
        self._last_message_id_logs[message_id] = log
        if len(self._last_message_id_logs) > 10:
            oldest_key = next(iter(self._last_message_id_logs))
            del self._last_message_id_logs[oldest_key]

    def get_log_by_id(self, message_id: int) -> str:
        return self._last_message_id_logs.get(message_id, "(NONE FOUND)")

    async def should_process_message(self, message: discord.Message) -> bool:
        if not(self.bot.user in message.mentions):
            return False

        if len(message.content) > ChatHandler.MAX_CHAT_CHARACTERS:
            emojis = ['🇹', '🇱', '🇩', '🇷']
            for emoji in emojis:
                await message.add_reaction(emoji)
            return False

        if self.rate_limiter.is_rate_limited(message.author.id):
            await message.reply(f"{KamiChan.Vocabulary.EMOJI_NO} :x: `You are being rate limited`")
            return False

        return True

    async def answer_eta_question_if_needed(self, message: discord.Message):
        debug_mode = message.content.endswith("--eta")
        query = message.content.removesuffix("--eta")

        classification_result = await preset_queries.classify_eta_question(query)

        if debug_mode:
            await message.reply(
f"""```
--eta (DEBUG MODE):
            
The query was: {query}
Classification result: {classification_result.finish_reason}
Similarity: {classification_result.similarity}
LLM response: {classification_result.llm_classification_json}
```"""
) 
        if classification_result.finish_reason == "is_eta_question":
            await message.reply(
                embed=discord.Embed(
                    title="No ETA!", 
                    description=f"Paper does not publish ETAs for releases, or estimates based on previous versions. Every version is different! Please stay tuned for new announcements."
                )
            )
        
    # TODO: refactor this
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return

        if message.author.id == 688858519486857252 and "sync command tree" in message.content:
            print("Synced command tree")
            await self.bot.tree.sync()

        if await self.answer_eta_question_if_needed(message) or message.content.endswith("--eta"):
            return

        logs = message.content.endswith("--l")
        
        if logs:
            sanitized_msg = ""
            try:
                sanitized_msg = message.content.strip().replace("--l", "")
                message_id = int(sanitized_msg)
                log_file = io.StringIO(self.get_log_by_id(message_id))
                await message.reply(
                    content=f"Verbose logs for message ID {message_id} attached (only last 10 are stored)", 
                    files=[discord.File(log_file, filename="verbose_log.txt")]
                )
            except ValueError:
                await message.reply(f":x: Expected a message ID before --l, not '{sanitized_msg}'")
                return

        if not await self.should_process_message(message):
            return

        self.rate_limiter.register_request(message.author.id)
        await self.ai_bot.memorize_short_term(message)
        await self.vector_db_conn.add_messages([message])
        reply = await message.reply(f"-# {KamiChan.Vocabulary.EMOJI_UWU} {BOT_NAME} is typing...")
        verbose = message.content.endswith("--v")
        
        try:
            disclaimer = f"-# Unofficial bot. FICTITIOUS AI-generated content. | [Learn more.](https://discord.com/channels/532557135167619093/1192649325709381673/1196285641978302544)"
            resp = DiscordBotResponse(self.ai_bot, verbose)
            resp_str = await resp.create_or_fallback(message, ["google/gemini-flash-1.5", "qwen/qwen-2-72b-instruct", "meta-llama/llama-3-70b-instruct"])
            reply_msg = await reply.edit(content=resp_str)
            await self.ai_bot.memorize_short_term(reply_msg)
            await self.vector_db_conn.add_messages([reply_msg])
            if resp.verbose:
                log_file = io.StringIO(resp.verbose_log)
                reply = await reply.edit(
                    content=resp_str, 
                    attachments=[discord.File(log_file, filename="verbose_log.txt")]
                )
            await reply.edit(content=f"{resp_str}\n{disclaimer}") # TODO: clean up logic so that this isn't needed
            self.cache_log(reply.id, resp.verbose_log)
        except Exception as e:
            await self.ai_bot.forget_short_term(message)
            await self.ai_bot.forget_short_term(reply)
            traceback.print_exc()
            await reply.edit(content=f"Sorry, there was an error!! {KamiChan.Vocabulary.EMOJI_DESPAIR} ```{str(e)}```")