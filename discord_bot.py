import discord
from discord.ext import commands

INSTANCE = commands.Bot(command_prefix='paper!', intents=discord.Intents.default())
