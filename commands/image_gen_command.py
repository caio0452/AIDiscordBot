import discord
import json
import io
import openai
import requests
import httpx
import traceback
from discord import app_commands
from discord.ext import commands
from util.rate_limits import RateLimit, RateLimiter
from bot_workflow.profile_loader import Profile

class ImageGenCommand(commands.Cog):
    def __init__(self, discord_bot: commands.Bot) -> None:
        self.discord_bot = discord_bot
        self.bot_profile = ai_bot_profile
        self.image_gen_rate_limiter = RateLimiter(RateLimit(n_messages=3, seconds=60))

    async def _is_blocked_prompt(self, prompt: str) -> bool:
        blocked_words = ["nsfw", "naked", "bikini", "lingerie", "sexy", "penis", "fuck", "murder", "blood"]

        for word in blocked_words:
            if word in prompt:
                return True

        response = await self.openai_client.chat.completions.create(
            # messages = PROMPT TODO
            model="openai/gpt-4o-mini",
            max_tokens=64,
            temperature=0,
            response_format={ "type": "json_object" }
        )
        response_data = json.loads(response.choices[0].message.content)

        if response_data["mentions_sexual_content"] or response_data["violent_content"] == "high" or response_data["graphic_content"] == "high":
            return True
        
        return False

    async def _fal_ai_request_image(self, request: str):
        url = "https://fal.run/fal-ai/realistic-vision"

        headers = {
            "Authorization": f"Key {self.bot_profile.fal_image_gen_config.}",
            "Content-Type": "application/json",
        }
        data = {
            "prompt": request,
            "model_name": "SG161222/RealVisXL_V4.0",
            "negative_prompt": "Bad anatomy, ugly, low quality, low detail, blurry",
            "enable_safety_checker": True,
            "num_images": 1
        }

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(url, headers=headers, json=data)
            return response

    @app_commands.command(
        name="generate_image", 
        description="Generate an image"
    )
    async def generate_image(self, interaction: discord.Interaction, query: str) -> None:
        user_id = interaction.user.id
        await interaction.response.defer()

        try:
            if await self._is_blocked_prompt(query):
                await interaction.followup.send(":x: Prompt flagged")
                return
            
            if self.image_gen_rate_limiter.is_rate_limited(user_id):
                await interaction.followup.send(":x: You are being rate limited (3 / min)")
                return
            
            self.image_gen_rate_limiter.register_request(user_id)
            req = await self._fal_ai_request_image(query)
            json_data = req.content.decode('utf-8')
            data = json.loads(json_data)

            image_url1 = data["images"][0]["url"]
            file1 = discord.File(io.BytesIO(requests.get(image_url1).content), filename="image1.png")

            await interaction.followup.send(
                f"`PROMPT:` **{query}**", file=file1
            )
        except Exception as e:
            traceback.print_exc()
            await interaction.followup.send(
                f":x: There was error generating the image: {str(e)}")