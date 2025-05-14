import discord
import logging
import core.util.logging_setup as logs

from discord.ext import commands
from core.bot_workflow.ai_bot import CustomBotData
from core.bot_workflow.profile_loader import Profile
from core.ai_apis.providers import ProviderDataStore
from core.util.environment_vars import get_environment_var
from core.bot_workflow.discord_chat_handler import DiscordChatHandler
from core.bot_workflow.knowledge import KnowledgeIndex, LongTermMemoryIndex

from commands.sync_command_tree import SyncCommand
from commands.image_gen_command import ImageGenCommand

logs.setup()

class DiscordBot:
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        self.bot = commands.Bot(command_prefix='r!', intents=intents)
        self.profile = Profile.from_file("profile.json")
        self.bot.event(self.on_ready)

    def run(self):
        bot_token = get_environment_var('AI_BOT_TOKEN', required=True)
        self.bot.run(bot_token)

    async def setup_chatbot(self):
        embeddings_provider = self.profile.providers["EMBEDDINGS"]
        self.knowledge = await KnowledgeIndex.from_provider(embeddings_provider)
        if self.profile.options.enable_long_term_memory:
            self.long_term_memory: LongTermMemoryIndex | None = await LongTermMemoryIndex.from_provider(embeddings_provider)
        else:
            self.long_term_memory = None

        provider_list = [self.profile.providers[k] for k, v in self.profile.providers.items()]
        provider_store = ProviderDataStore(
            providers=provider_list
        ) # TODO: There should be required providers
        if self.bot.user is None:
            raise RuntimeError("Could not initialize bot: bot user is None")
        await self.bot.add_cog(DiscordChatHandler(
            discord_bot=self.bot, 
            ai_bot_data=CustomBotData(
                name=self.profile.options.botname, 
                profile=self.profile, 
                provider_store=provider_store,
                long_term_memory=self.long_term_memory,
                knowledge=self.knowledge,
                discord_bot_id=self.bot.user.id,
                memory_length=50
            )
        ))

    async def setup_commands(self):
        # await self.bot.add_cog(SearchCommand(bot=self.bot,conn=conn))
        # await self.bot.add_cog(FindClosePreset(presets_manager=await preset_queries.manager(OAICompatibleProviderData(embeddings_client)), bot=self.bot))
        await self.bot.add_cog(SyncCommand(bot=self.bot))
        # await self.bot.add_cog(TranslateCommand(bot=self.bot))
        # await self.bot.add_cog(RewriteCommand(bot=self.bot))
        
        if bot.profile.fal_image_gen_config.enabled:
            await self.bot.add_cog(ImageGenCommand(discord_bot=self.bot, bot_profile=self.profile, fal_config=bot.profile.fal_image_gen_config))
        else:
            logging.info("Image generation using FAL.AI is disabled")
        pass

    async def on_ready(self):
        logging.info("Setting up commands...")
        await self.setup_commands()
        logging.info("Creating chatbot...")
        await self.setup_chatbot()
        logging.info("Indexing knowledge...")
        await self.knowledge.index_from_folder("brain_content/knowledge")
        logging.info(f'Logged in as {self.bot.user}')

bot = DiscordBot()
bot.run()