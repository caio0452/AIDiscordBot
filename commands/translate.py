from discord import app_commands
import openai
import providers
import discord

from discord.ext import commands
from ai import OAICompatibleProvider

class TranslateCommand(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        translation_provider = providers.get_provider_by_name("TRANSLATION")
        self._translation_client = None

        if translation_provider is None:
            print("Missing TRANSLATION provider")
        else:
            self._translation_client = openai.AsyncOpenAI(
                api_key=translation_provider.api_key, 
                base_url=translation_provider.api_base, 
                timeout=15
            )

    @app_commands.command(
        name="translate", 
        description="Translate text using an LLM"
    )
    async def translate(self, interaction: discord.Interaction, *, text: str) -> None:
        await interaction.response.defer()
        if self._translation_client is None:
            await interaction.followup.send(":x: Missing TRANSLATION provider")
            return

        translated_text = await self.call_llm_translate(text)
        await interaction.followup.send(translated_text)
    
    async def call_llm_translate(self, text: str) -> str:
        resp = await self._translation_client.chat.completions.create(
            messages=OAICompatibleProvider.system_msg(
                f'Translate to english the text between the <TEXT> tag. <TEXT>{text}</TEXT>.' \
                 'Reply with just a JSON containing {"translation": "[insert translated text here]"}'),
            model="gpt-3.5-turbo",
            max_tokens=1000,
            response_format={ "type": "json_object"}
        )
        return f"`TRANSLATION:` {resp.choices[0].message.content}"