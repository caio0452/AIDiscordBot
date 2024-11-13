import discord

from util import bot_config
from discord.ext import commands
from bot_workflow.ai_bot import CustomBotData
from bot_workflow.vector_db import VectorDatabase
from bot_workflow.knowledge import KnowledgeIndex
from ai_apis.providers import ProviderDataStore
from bot_workflow.personality_loader import PersonalityLoader
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
        embeddings_provider = self.ai_personality.providers["EMBEDDINGS"]
        self.vector_db = VectorDatabase(embeddings_provider)
        self.knowledge = KnowledgeIndex(embeddings_provider)
        self.bot.event(self.on_ready)

    def run(self):
        self.bot.run(bot_config.BOT_TOKEN)

    async def setup_chatbot(self):
        provider_list = [self.ai_personality.providers[k] for k, v in self.ai_personality.providers.items()]
        provider_store = ProviderDataStore(
            providers=provider_list
        ) # TODO: There should be required providers
        personality = PersonalityLoader("personality.json").load_personality()
        await self.bot.add_cog(DiscordChatHandler(
            discord_bot=self.bot, 
            ai_bot_data=CustomBotData(
                name="Kami-Chan", 
                vector_db=self.vector_db, 
                personality=personality, 
                provider_store=provider_store, 
                knowledge=self.knowledge,
                discord_bot_id=self.bot.user.id
            )
        ))

    async def setup_commands(self):
        # await self.bot.add_cog(SearchCommand(bot=self.bot,conn=conn))
        # await self.bot.add_cog(FindClosePreset(presets_manager=await preset_queries.manager(OAICompatibleProviderData(embeddings_client)), bot=self.bot))
        # await self.bot.add_cog(SyncCommand(bot=self.bot, allowed_user_id=688858519486857252))
        # await self.bot.add_cog(TranslateCommand(bot=self.bot))
        # await self.bot.add_cog(RewriteCommand(bot=self.bot))
        
        # if parameters.FAL_AI_KEY == "":
        #    print("No Fal.AI key specified, image generation will be disabled")
        # else:
        #    await self.bot.add_cog(ImageGenCommand(bot=self.bot))
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




