import discord

from util import bot_config
from discord.ext import commands
from bot_workflow.ai_bot import CustomBotData
from ai_apis.providers import ProviderDataStore
from bot_workflow.profile_loader import ProfileLoader
from bot_workflow.knowledge import KnowledgeIndex, LongTermMemoryIndex
from bot_workflow.discord_chat_handler import DiscordChatHandler

# from commands.rewrite import RewriteCommand
# from commands.translate import TranslateCommand
# from commands.search_command import SearchCommand
from commands.sync_command_tree import SyncCommand
from commands.image_gen_command import ImageGenCommand

class DiscordBot:
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        self.bot = commands.Bot(command_prefix='paper!', intents=intents)
        self.profile = ProfileLoader("profile.json").load_profile()
        embeddings_provider = self.profile.providers["EMBEDDINGS"]
        self.knowledge = KnowledgeIndex(embeddings_provider)
        self.long_term_memory = LongTermMemoryIndex(embeddings_provider)
        self.bot.event(self.on_ready)

    def run(self):
        self.bot.run(bot_config.BOT_TOKEN)

    async def setup_chatbot(self):
        provider_list = [self.profile.providers[k] for k, v in self.profile.providers.items()]
        provider_store = ProviderDataStore(
            providers=provider_list
        ) # TODO: There should be required providers
        profile = ProfileLoader("profile.json").load_profile()
        await self.bot.add_cog(DiscordChatHandler(
            discord_bot=self.bot, 
            ai_bot_data=CustomBotData(
                name=profile.botname, 
                profile=profile, 
                provider_store=provider_store,
                long_term_memory=self.long_term_memory,
                knowledge=self.knowledge,
                discord_bot_id=self.bot.user.id
            )
        ))

    async def setup_commands(self):
        # await self.bot.add_cog(SearchCommand(bot=self.bot,conn=conn))
        # await self.bot.add_cog(FindClosePreset(presets_manager=await preset_queries.manager(OAICompatibleProviderData(embeddings_client)), bot=self.bot))
        await self.bot.add_cog(SyncCommand(bot=self.bot, allowed_user_id=688858519486857252))
        # await self.bot.add_cog(TranslateCommand(bot=self.bot))
        # await self.bot.add_cog(RewriteCommand(bot=self.bot))
        
        if bot.profile.fal_image_gen_config.enabled:
            # await self.bot.add_cog(ImageGenCommand(bot=self.bot)) - TODO: reimplement
            pass
        else:
            print("Image generation using FAL.AI is disabled")
        pass

    async def on_ready(self):
        print("Setting up commands...")
        await self.setup_commands()
        print("Creating chatbot...")
        await self.setup_chatbot()
        print("Indexing knowledge...")
        await self.knowledge.index_from_folder("knowledge")
        print(f'Logged in as {self.bot.user}')

bot = DiscordBot()
bot.run()




