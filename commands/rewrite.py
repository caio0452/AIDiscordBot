from discord import app_commands
import openai
import providers
import discord

from discord.ext import commands
from ai import OAICompatibleProvider

class RewriteCommand(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        rewrite_provider = providers.get_provider_by_name("DEFAULT")
        self._rewrite_client: openai.AsyncOpenAI | None = None

        if rewrite_provider is None:
            print("Missing DEFAULT provider")
        else:
            self._rewrite_client = openai.AsyncOpenAI(
                api_key=rewrite_provider.api_key, 
                base_url=rewrite_provider.api_base, 
                timeout=15
            )

    @app_commands.command(
        name="rewrite", 
        description="Rewrite text using an LLM"
    )
    async def rewrite(self, interaction: discord.Interaction, *, text: str) -> None:
        msg = await interaction.followup.send(content="This wil take a while, querying multiple AIs...")
        await interaction.response.defer()

        if self._rewrite_client is None:
            await interaction.followup.send(":x: Missing DEFAULT provider")
            return

        rewritten_text = await self.call_llm_rewrite(text)
        await msg.edit(rewritten_text)
    
    async def call_llm_rewrite(self, text: str) -> str:
        models = ["microsoft/phi-3-medium-128k-instruct", "anthropic/claude-3-haiku:beta", "meta-llama/llama-3.1-70b-instruct"]
        rewrites = []

        for model in models:
            try:
                resp = await self._rewrite_client.chat.completions.create(
                    messages=[OAICompatibleProvider.system_msg(
                        f'Rewrite the text between the <TEXT> tags in a different way. Avoid using the same words, and change the text structure, but keep the length. <TEXT>{text}</TEXT>.' \
                        'Reply with just a JSON containing {"rewrite": "[insert reworded text here]"}')],
                    model=model,
                    max_tokens=4000,
                    response_format={"type": "json_object"}
                )
                rewrite = f"**{model}**: {resp.choices[0].message.content}"
                rewrites.append(rewrite)
            except Exception as e:
                rewrites.append(f"**{model}**: Error - {str(e)}")

        return "**REWRITES:**\n" + "\n\n".join(rewrites)