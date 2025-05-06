import discord
from discord import app_commands
from discord.ext import commands

class SyncCommand(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
    
    @app_commands.command(
        name="sync", 
        description="Sync the command tree"
    )
    @commands.is_owner()
    async def sync_commands(self, interaction: discord.Interaction) -> None:
        await self.bot.tree.sync()
        await interaction.followup.send("Command tree synced.")