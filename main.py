import openai
from AIDiscordBot.ai import OAICompatibleProvider
import parameters
import preset_queries

from commands.image_gen_command import ImageGenCommand
from commands.search_command import SearchCommand
from commands.find_close_preset import FindClosePreset
from commands.sync_command_tree import SyncCommand

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
json_file_path = "queries.json"

async def setup_commands():
    await bot.add_cog(ChatHandler(bot=bot, db_connection=conn))
    await bot.add_cog(SearchCommand(bot=bot,conn=conn))
    await bot.add_cog(FindClosePreset(presets_manager=await preset_queries.manager(OAICompatibleProvider(embeddings_client)), bot=bot))
    await bot.add_cog(SyncCommand(bot=bot, allowed_user_id=688858519486857252))
    if parameters.FAL_AI_KEY == "":
        print("No Fal.AI key specified, image generation will be disabled")
    else:
        await bot.add_cog(ImageGenCommand(bot=bot))

@bot.event
async def on_ready():
    await Knowledge(conn).start_indexing()
    await setup_commands()
    print(f'Logged in as {bot.user}')

bot.run(parameters.BOT_TOKEN)




