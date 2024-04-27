import openai
from commands.search_command import SearchCommand
import parameters

from chat_handler import ChatHandler
from knowledge import Knowledge
from vector_db import QdrantVectorDb
import discord_bot
import parameters

OPENAI_EMBEDDINGS_VECTOR_SIZE = 3072 # TODO: varies with model
openai_client = openai.AsyncOpenAI(api_key=parameters.api_key)

db_client = QdrantVectorDb(
    parameters.qdrant_url,
    openai_client,
    int(parameters.qdrant_port),
    OPENAI_EMBEDDINGS_VECTOR_SIZE
)
conn = db_client.connect()
bot = discord_bot.INSTANCE

async def setup_commands():
    await bot.add_cog(ChatHandler(bot=bot, openai_client=openai_client, db_connection=conn))
    await bot.add_cog(SearchCommand(bot=bot,conn=conn))

@bot.event
async def on_ready():
    await Knowledge(conn).start_indexing()
    await setup_commands()
    print(f'Logged in as {bot.user}')

bot.run(parameters.bot_token)




