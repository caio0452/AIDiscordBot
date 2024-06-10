import discord
from discord import app_commands
from discord.ext import commands
from preset_queries import PresetQueryManager

class FindClosePreset(commands.Cog):
  def __init__(self, presets_manager:  PresetQueryManager,bot: commands.Bot) -> None:
    self.bot = bot
    self.presets_manager = presets_manager

  @app_commands.command(
        name="findclosepreset", 
        description="Search Paper-Chan's known canned responses"
    )
  async def on_find_close_preset_command(self, interaction: discord.Interaction, query: str) -> None:
    await interaction.response.defer()
    resp = "" 
    for preset in self.presets_manager.all_preset_queries:
      resp += preset.preset_question + "\n"
    await interaction.response.send_message(content=f"Here's the preset answers that I know and are similar to your query: ```{resp}```")