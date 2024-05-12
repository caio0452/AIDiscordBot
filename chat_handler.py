import discord
import traceback
import io
from discord.ext import commands
from vector_db import QdrantVectorDbConnection
from rate_limits import RateLimit, RateLimiter
from kami_chan.kami_chan import KamiChan

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
        self.paper_chan = KamiChan("Kami-Chan", db_connection, self.bot.user.id)

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
        try:
            str_response = await self.paper_chan.respond_to_query(message)
            disclaimer = "<:unofficial:1233866785862848583><:unofficial_1:1233866787314073781><:unofficial_2:1233866788777754644>  | [Learn more.](https://discord.com/channels/532557135167619093/1192649325709381673/1196285641978302544)"
            reply_msg = await reply.edit(content=str_response + "\n" + disclaimer)
            await self.paper_chan.memorize_short_term(reply_msg)
            await self.vector_db_conn.add_messages([reply_msg])
        except Exception as e:
            error_traceback = ''.join(traceback.format_exception(e))
            error_file = io.StringIO(error_traceback)
            await reply.edit(content="Sorry, there was error!!", attachments=[discord.File(error_file, filename="error_traceback.txt")])