import discord
import json
import io
import openai
import requests
import httpx
import parameters
import providers
from discord import app_commands
from discord.ext import commands
from rate_limits import RateLimiter, RateLimit

class ImageGenCommand(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.image_gen_rate_limiter = RateLimiter(RateLimit(n_messages=3, seconds=60))
        provider = providers.get_provider_by_name("IMAGE_GEN_MODERATOR_CLIENT")
        self.openai_client = openai.AsyncOpenAI(api_key=provider.api_key, base_url=provider.api_base)
        self.fal_ai_key = parameters.FAL_AI_KEY

    async def _is_blocked_prompt(self, prompt: str) -> bool:
        blocked_words = ["nsfw", "naked", "bikini", "lingerie", "sexy", "penis", "fuck", "murder", "blood"]

        for word in blocked_words:
            if word in prompt:
                return True
    
        response = await self.openai_client.chat.completions.create(
            messages = [
            {
                'role': 'system',
                'content': 'You are a query classifier. Reply with a JSON object indicating the presence of ANY mention of sexual content and the levels of violent and graphic content. The JSON must have format {"mentions_sexual_content": false or true, "violent_content": "low" or "medium" or "high", "graphic_content": "low" or "medium" or "high"},'
            },
            {
                'role': 'user',
                'content': 'Anime girl in bikini'
            },
            {
                'role': 'assistant',
                'content': '{"mentions_sexual_content": true, "violent_content": "low", "graphic_content": "medium"}'
            },
            {
                'role': 'user',
                'content': 'A photo of a house exploding'
            },
            {
                'role': 'assistant',
                'content': '{"mentions_sexual_content": false, "violent_content": "medium", "graphic_content": "medium"}'
            },
            {
                'role': 'user',
                'content': 'A photo of a woman sitting on a sofa on her house, HD, 4K, detailed'
            },
            {
                'role': 'assistant',
                'content': '{"mentions_sexual_content": false, "violent_content": "low", "graphic_content": "low"}'
            },
            {
                'role': 'user',
                'content': 'drawing of sexy man'
            },
            {
                'role': 'assistant',
                'content': '{"mentions_sexual_content": true, "violent_content": "low", "graphic_content": "medium"}'
            },
            {
                'role': 'user',
                'content': 'A picture without any sexual content in it'
            },
            {
                'role': 'assistant',
                'content': '{"mentions_sexual_content": true, "violent_content": "low", "graphic_content": "low"}'
            },
            {
                'role': 'user',
                'content': 'No nudity allowed'
            },
            {
                'role': 'assistant',
                'content': '{"mentions_sexual_content": true, "violent_content": "low", "graphic_content": "low"}'
            },
            {
                'role': 'user',
                'content': 'A person with a painful headache'
            },
            {
                'role': 'assistant',
                'content': '{"mentions_sexual_content": false, "violent_content": "low", "graphic_content": "low"}'
            },
            {
                'role': 'user',
                'content': 'A terrorist blowing up people'
            },
            {
                'role': 'assistant',
                'content': '{"mentions_sexual_content": false, "violent_content": "high", "graphic_content": "high"}'
            },
            {
                'role': 'user',
                'content': 'A woman [not] dressed'
            },
            {
                'role': 'assistant',
                'content': '{"mentions_sexual_content": true, "violent_content": "low", "graphic_content": "high"}'
            },
            {
                'role': 'user',
                'content': 'A male individual "without" clothing'
            },
            {
                'role': 'assistant',
                'content': '{"mentions_sexual_content": true, "violent_content": "low", "graphic_content": "high"}'
            },
            {
                'role': 'user',
                'content': prompt
            }
        ],
            model="gpt-3.5-turbo",
            max_tokens=64,
            temperature=0,
            response_format={ "type": "json_object" }
        )
        response_data = json.loads(response.choices[0].message.content)

        if response_data["mentions_sexual_content"] or response_data["violent_content"] == "high" or response_data["graphic_content"] == "high":
            return True
        
        return False

    async def _fal_ai_request_image(self, request: str):
        url = "https://fal.run/fal-ai/pixart-sigma"

        headers = {
            "Authorization": f"Key {self.fal_ai_key}",
            "Content-Type": "application/json",
        }
        data = {
            "prompt": request,
            "negative_prompt": "Bad anatomy, ugly, low quality, low detail, blurry",
            "enable_safety_checker": True,
            "num_images": 1
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=data)
            return response

    @app_commands.command(
        name="generate_image", 
        description="Generate an image"
    )
    async def generate_image(self, interaction: discord.Interaction, query: str) -> None:
        user_id = interaction.user.id
        await interaction.response.defer()

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

        try:
            image_url1 = data["images"][0]["url"]
            file1 = discord.File(io.BytesIO(requests.get(image_url1).content), filename="image1.png")

            await interaction.followup.send(
                f"`PROMPT:` **{query}**", file=file1
            )

        except KeyError:
            await interaction.followup.send(
                f":x: Error generating image.\n```json\n{json_data[0:1700]}```"
            )