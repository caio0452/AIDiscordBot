import openai
import environment_vars

from chat_handler import ChatHandler
from knowledge import Knowledge
from vector_db import QdrantVectorDb
import discord_bot
import environment_vars

# TODO: varies with model
OPENAI_EMBEDDINGS_VECTOR_SIZE = 3072

openai_client = openai.AsyncOpenAI(api_key=environment_vars.api_key)

db_client = QdrantVectorDb(
    environment_vars.qdrant_url,
    openai_client,
    int(environment_vars.qdrant_port),
    OPENAI_EMBEDDINGS_VECTOR_SIZE
)
conn = db_client.connect()

bot = discord_bot.INSTANCE

# TODO: move commands to cogs
@bot.tree.command(
    name="faq",
    description="Testing a FAQ"
)
async def faq_command(interaction, query: str):
    results = await conn.query_qa_knowledge(query)
    found_text = ""
    for i, result in enumerate(results):
        found_text += f"`<{i}>`" + result.payload[0:150] + "... \n `SCORE: " + "{:.2f}".format(result.score * 100) + "/100`\n"

    await interaction.response.send_message("**Here's the relevant things I found in my knowledge database:**\n" + found_text[0:1900])

@bot.event
async def on_ready():
    await Knowledge(conn).start_indexing()
    await bot.add_cog(ChatHandler(
            bot=bot,
            openai_client=openai_client,
            db_connection=conn,
        )
    )
    print(f'Logged in as {bot.user}')

@bot.tree.command(
    name="search",
    description="Search the available knowledge"
)
async def search_command(interaction, query: str):
    results = await self.conn.query_relevant_knowledge(query)
    found_text = ""
    for i, result in enumerate(results):
        found_text += f"`<{i}>`" + result.payload[0:150] + "... \n `SCORE: " + "{:.2f}".format(result.score * 100) + "/100`\n"

    await interaction.response.send_message("**Here's the relevant things I found in my knowledge database:**\n" + found_text[0:1900])

bot.run(environment_vars.bot_token)




