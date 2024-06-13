import discord
from discord.ext import commands

intents = discord.Intents.default()
intents.message_content = True

INSTANCE = commands.Bot(command_prefix='paper!', intents=intents)
