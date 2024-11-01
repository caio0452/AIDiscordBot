import discord

from util import bot_config
from discord.ext import commands
from bot_workflow.vector_db import VectorDatabase
from AIDiscordBot.ai_apis.providers import ProviderDataStore
from AIDiscordBot.bot_workflow.personality_loader import PersonalityLoader
from bot_workflow.discord_chat_handler import DiscordChatHandler
# from commands.rewrite import RewriteCommand
# from commands.translate import TranslateCommand
# from commands.search_command import SearchCommand
# from commands.sync_command_tree import SyncCommand
# from commands.image_gen_command import ImageGenCommand

class DiscordBot:
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        self.bot = commands.Bot(command_prefix='paper!', intents=intents)
        self.ai_personality = PersonalityLoader("personality.json").load_personality()
        self.vector_db = VectorDatabase(self.ai_personality.providers["EMBEDDINGS"])
        self.bot.event(self.on_ready)

    def run(self):
        self.bot.run(bot_config.BOT_TOKEN)

    async def setup_commands(self):
        provider_list = [self.ai_personality.providers[k] for k, v in self.ai_personality.providers.items()]
        provider_store = ProviderDataStore(
            providers=provider_list
        ) # TODO: There should be required providers
        await self.bot.add_cog(DiscordChatHandler(
            bot=self.bot, 
            provider_store=provider_store, 
            vector_database=self.vector_db
        ))
        # await self.bot.add_cog(SearchCommand(bot=self.bot,conn=conn))
        # await self.bot.add_cog(FindClosePreset(presets_manager=await preset_queries.manager(OAICompatibleProviderData(embeddings_client)), bot=self.bot))
        # await self.bot.add_cog(SyncCommand(bot=self.bot, allowed_user_id=688858519486857252))
        # await self.bot.add_cog(TranslateCommand(bot=self.bot))
        # await self.bot.add_cog(RewriteCommand(bot=self.bot))
        
        # if parameters.FAL_AI_KEY == "":
        #    print("No Fal.AI key specified, image generation will be disabled")
        # else:
        #    await self.bot.add_cog(ImageGenCommand(bot=self.bot))

    async def on_ready(self):
        await self.setup_commands()
        print(f'Logged in as {self.bot.user}')

bot = DiscordBot()
bot.run()




