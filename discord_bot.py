import discord
from discord.ext import commands
from discord import app_commands

INSTANCE = commands.Bot(command_prefix='paper!', intents=discord.Intents.default())
