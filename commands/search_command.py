import discord
from discord import app_commands
from discord.ext import commands
from vector_db import QdrantVectorDbConnection

class SearchCommand(commands.Cog):
  def __init__(self, bot: commands.Bot, conn: QdrantVectorDbConnection) -> None:
    self.bot = bot
    self.conn = conn
    
  @app_commands.command(
        name="dbsearch", 
        description="Search Paper-Chan's knowledge database"
    )
  async def on_faq_command(self, interaction: discord.Interaction, query: str) -> None:
    results = await self.conn.query_relevant_knowledge(query)
    found_text = ""
    for i, result in enumerate(results):
        result_text = result.payload[0:150]
        score = {100 * result.score}
        found_text += f"`<{i}>` {result_text}\n `SCORE: {score}/100`\n"

    await interaction.response.send_message("**Here's the relevant things I found in my knowledge database:**\n" + found_text[0:1900])