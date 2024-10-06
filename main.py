import discord
import parameters

from discord.ext import commands
from chat_handler import ChatHandler
from commands.rewrite import RewriteCommand
from commands.translate import TranslateCommand
from commands.search_command import SearchCommand
from commands.sync_command_tree import SyncCommand
from commands.image_gen_command import ImageGenCommand
from providers import ProviderStore
from vector_db import VectorDatabase

class DiscordBot:
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        self.bot = commands.Bot(command_prefix='paper!', intents=intents)
        self.provider_store = ProviderStore.load_from_json("providers.json")
        self.vector_db = VectorDatabase(self.provider_store.get_provider_by_name("EMBEDDDINGS"))

    def run(self):
        self.bot.run(parameters.BOT_TOKEN)

    async def setup_commands(self):
        await self.bot.add_cog(ChatHandler(
            bot=self.bot, 
            provider_store=self.provider_store, 
            vector_database=self.vector_db
        ))
        # await self.bot.add_cog(SearchCommand(bot=self.bot,conn=conn))
        # await self.bot.add_cog(FindClosePreset(presets_manager=await preset_queries.manager(OAICompatibleProvider(embeddings_client)), bot=self.bot))
        # await self.bot.add_cog(SyncCommand(bot=self.bot, allowed_user_id=688858519486857252))
        # await self.bot.add_cog(TranslateCommand(bot=self.bot))
        # await self.bot.add_cog(RewriteCommand(bot=self.bot))
        
        # if parameters.FAL_AI_KEY == "":
        #    print("No Fal.AI key specified, image generation will be disabled")
        # else:
        #    await self.bot.add_cog(ImageGenCommand(bot=self.bot))
        self.bot.event(self.on_ready)

    async def on_ready(self):
        await self.setup_commands()
        print(f'Logged in as {self.bot.user}')

bot = DiscordBot()
bot.run()




