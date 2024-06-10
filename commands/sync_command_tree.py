import discord
from discord import app_commands
from discord.ext import commands

class SyncCommand(commands.Cog):
    def __init__(self, bot: commands.Bot, *, allowed_user_id: int) -> None:
        self.bot = bot
        self.allowed_user_id = allowed_user_id
    
    @commands.command(
        name="sync", 
        description="Sync the command tree"
    )
    async def sync_commands(self, ctx: commands.Context) -> None:
        if ctx.author.id != self.allowed_user_id:
            await ctx.send("You do not have permission to run this command.")
            return
        
        await self.bot.tree.sync()
        await ctx.send("Command tree synced.")