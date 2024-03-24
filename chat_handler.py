import discord
import openai
from discord.ext import commands
from vector_db import QdrantVectorDbConnection
from rate_limits import RateLimit, RateLimiter
from paper_chan.paper_chan import PaperChan

class ChatHandler(commands.Cog):
    MAX_CHAT_CHARACTERS = 500

    def __init__(self, openai_client: openai.AsyncOpenAI, bot: commands.Bot, db_connection: QdrantVectorDbConnection):
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
        self.paper_chan = PaperChan("Paper-Chan", openai_client, db_connection, self.bot.user.id)

    async def should_process_message(self, message: discord.Message) -> bool:
        if message.author.bot or not(self.bot.user in message.mentions):
            return False

        if len(message.content) > ChatHandler.MAX_CHAT_CHARACTERS:
            emojis = ['🇹', '🇱', '🇩', '🇷']
            for emoji in emojis:
                await message.add_reaction(emoji)
            return False

        if self.rate_limiter.is_rate_limited(message.author.id):
            await message.reply("<:Paperno:1022991562810077274> :x: `You are being rate limited`")
            return False

        return True

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if not await self.should_process_message(message):
            return

        self.rate_limiter.register_request(message.author.id)
        await self.paper_chan.memorize_short_term(message)
        await self.vector_db_conn.add_messages([message])
        reply = await message.reply("<:paperUwU:1018366709658308688> Paper-Chan is typing...")
        str_response = await self.paper_chan.respond_to_query(message)
        reply_msg = await reply.edit(content=str_response)
        await self.paper_chan.memorize_short_term(reply_msg)
        await self.vector_db_conn.add_messages([reply_msg])