import openai
from commands.search_command import SearchCommand
import parameters

from chat_handler import ChatHandler
from knowledge import Knowledge
from vector_db import QdrantVectorDb
import discord_bot
import parameters
import providers

OPENAI_EMBEDDINGS_VECTOR_SIZE = 3072 # TODO: varies with model

embeddings_provider = providers.get_provider_by_name("EMBEDDINGS_PROVIDER")
embeddings_client = openai.AsyncOpenAI(api_key=embeddings_provider.api_key, base_url=embeddings_provider.api_base)

db_client = QdrantVectorDb(
    parameters.QDRANT_URL,
    embeddings_client,
    int(parameters.QDRANT_PORT),
    OPENAI_EMBEDDINGS_VECTOR_SIZE
)
conn = db_client.connect()
bot = discord_bot.INSTANCE

async def setup_commands():
    await bot.add_cog(ChatHandler(bot=bot, db_connection=conn))
    await bot.add_cog(SearchCommand(bot=bot,conn=conn))

@bot.event
async def on_ready():
    await Knowledge(conn).start_indexing()
    await setup_commands()
    print(f'Logged in as {bot.user}')

bot.run(parameters.BOT_TOKEN)




