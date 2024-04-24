import openai
from commands.search_command import SearchCommand
import environment_vars

from chat_handler import ChatHandler
from knowledge import Knowledge
from vector_db import QdrantVectorDb
import discord_bot
import environment_vars

OPENAI_EMBEDDINGS_VECTOR_SIZE = 3072 # TODO: varies with model
openai_client = openai.AsyncOpenAI(api_key=environment_vars.api_key)

db_client = QdrantVectorDb(
    environment_vars.qdrant_url,
    openai_client,
    int(environment_vars.qdrant_port),
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

bot.run(environment_vars.bot_token)




