import openai
import parameters

from commands.image_gen_command import ImageGenCommand
from commands.search_command import SearchCommand
from commands.find_close_preset import FindClosePreset
from commands.sync_command_tree import SyncCommand

from preset_queries import PresetQueryManager, PresetQuery, KeywordMatcher, EmbeddingSimilarityMatcher
from ai import OAICompatibleProvider

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

async def build_preset_queries_db() -> PresetQueryManager: 
    print("Building query db...")
    embeddings_oai_wrapper = OAICompatibleProvider(embeddings_client)
    presets_manager = PresetQueryManager(embeddings_oai_wrapper)

    await presets_manager.add_query(
        PresetQuery(
            preset_question="How to fix slow chunk loading in Minecraft?", 
            embedding=await embeddings_oai_wrapper.vectorize("How to fix slow chunk loading in Minecraft?"),
            required_matchers=[KeywordMatcher(["chunk"]), EmbeddingSimilarityMatcher(await embeddings_oai_wrapper.vectorize("How to fix slow chunk loading in Minecraft?"), 0.5)]
        )
    )
    await presets_manager.add_query(
        PresetQuery(
            preset_question="Who is EterNity?", 
            embedding=await embeddings_oai_wrapper.vectorize("Who is EterNity?"),
            required_matchers=[KeywordMatcher(["who", "eter"]), EmbeddingSimilarityMatcher(await embeddings_oai_wrapper.vectorize("Who is EterNity?"), 0.5)]
        )
    )
    await presets_manager.add_query(
        PresetQuery(
            preset_question="How to disable chat reporting?", 
            embedding=await embeddings_oai_wrapper.vectorize("How to disable chat reporting?"),
            required_matchers=[KeywordMatcher(["chat", "rep", "moj"]), EmbeddingSimilarityMatcher(await embeddings_oai_wrapper.vectorize("How to disable chat reporting?"), 0.5)]
        )
    )
    print("Done")
    return presets_manager

async def setup_commands():
    await bot.add_cog(ChatHandler(bot=bot, db_connection=conn))
    await bot.add_cog(SearchCommand(bot=bot,conn=conn))
    await bot.add_cog(FindClosePreset(presets_manager=await build_preset_queries_db(), bot=bot))
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




