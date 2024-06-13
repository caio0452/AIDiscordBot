import discord
import io
import providers
import traceback
import numpy as np
import json

from openai import AsyncOpenAI
from discord.ext import commands
from vector_db import QdrantVectorDbConnection
from rate_limits import RateLimit, RateLimiter
from kami_chan.kami_chan import KamiChan, DiscordBotResponse

BOT_NAME = "Kami-Chan"

class ChatHandler(commands.Cog):
    MAX_CHAT_CHARACTERS = 500

    def __init__(self, bot: commands.Bot, db_connection: QdrantVectorDbConnection):
        self.bot: commands.Bot = bot
        self.vector_db_conn: QdrantVectorDbConnection = db_connection
        self.RECENT_MEMORY_LENGTH = 5
        self.rate_limiter = RateLimiter(
            RateLimit(n_messages=3, seconds=10),
            RateLimit(n_messages=10, seconds=60),
            RateLimit(n_messages=35, seconds=5 * 60),
            RateLimit(n_messages=100, seconds=2 * 3600),
            RateLimit(n_messages=250, seconds=8 * 3600)
        )
        self.ai_bot = KamiChan(BOT_NAME, db_connection, self.bot.user.id)

    # TODO: everything below this should probably should be mostly the bot's responsability
    async def should_process_message(self, message: discord.Message) -> bool:
        if message.author.bot or not(self.bot.user in message.mentions):
            return False

        if len(message.content) > ChatHandler.MAX_CHAT_CHARACTERS:
            emojis = ['🇹', '🇱', '🇩', '🇷']
            for emoji in emojis:
                await message.add_reaction(emoji)
            return False

        if self.rate_limiter.is_rate_limited(message.author.id):
            await message.reply(f"{KamiChan.Vocabulary.EMOJI_NO} :x: `You are being rate limited`")
            return False

        return True

    async def answer_eta_question_if_needed(self, message: discord.Message) -> bool:
        MIN_SIMILARITY = 0.5
        debug_mode = message.content.endswith("--eta")
        # 'updat' to catch update, updating, etc
        needed_keywords = ["eta", "when", "out", "will", "paper", ".", "release", "updat", "progress", "come"] 
        oai_client = AsyncOpenAI(api_key=providers.get_provider_by_name("EMBEDDINGS_PROVIDER").api_key)
        query = message.content.removeprefix("--eta")

        if not any([kw in message.content.lower() for kw in needed_keywords]):
            return False

        query_emb = await oai_client.embeddings.create(
            model="text-embedding-3-large", 
            input=message.content.lower()
        )
        reference_emb = await oai_client.embeddings.create(
            model="text-embedding-3-large", 
            input="Is there any ETA / estimate / progress on when 1.21 will release / come out?"
        )

        similarity = np.dot(query_emb.data[0].embedding, reference_emb.data[0].embedding)
        if similarity < MIN_SIMILARITY and not debug_mode:
            return False

        llm_classification_resp = await oai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{
                "role": "system", 
                "content": 
                """
                You classify queries that may possibly be asking for estimates (ETA - estimated time of arrival) or release dates for Paper 1.21.
                Given a query, return only a JSON containing :
                * wants_release_info: If the query aims to get info about the update, true or false
                * project_name: The name of the project if any, may be "none"
                * version: The version mentioned if any, may be "none" 

                Examples:
                Query: hi, do you know when 1.21 will be out?
                JSON: {"wants_release_info": true, "": project_name: "none", "version": "1.21"}
                Query: I would like to know when Paper will update
                JSON: {"wants_release_info": true, "": project_name: "Paper", "version": "none"}
                Query: vanilla 1.20 out when????
                JSON: {"wants_release_info": true, "": project_name: "vanilla", "version": "1.20"}
                Query: I hate it when people keep asking if 1.21 will come out
                JSON: {"wants_release_info": false, "": project_name: "none", "version": "1.21"}
                Query: man, i wish Paper would just hard fork so they can update to 1.21
                JSON: {"wants_release_info": false, "": project_name: "Paper", "version": "1.21"}
                Query: where is velocity 1.21??
                JSON: {"wants_release_info": false, "": project_name: "Velocity", "version": "1.21"}

                The query is now ((query)). Reply with just the corresponding JSON.
                JSON: 
                """.replace("((query))", query)
            }]
        )
        response_content = llm_classification_resp.choices[0].message.content
        response_json = json.loads(response_content)
        proj_name: str = response_json["project_name"]
        is_third_party_project = not any(proj in proj_name.lower() for proj in ["none", "paper", "velocity"])
        version = response_json["version"]

        if debug_mode:
            await message.reply(f"```SIMILARITY={similarity}\n\nCLASSIFICATION_DATA={response_content}```")

        if not response_json["wants_release_info"]:
            return False

        if version != "1.21" and version != "none":
            return False
        
        is_eta_question = not is_third_party_project

        if is_eta_question:
            await message.reply(
                embed=discord.Embed(
                    title="No ETA!", 
                    description=f"Paper does not publish ETAs for releases, or estimates based on previous versions. Every version is different! Please stay tuned for new announcements."
                )
            )

        return is_eta_question 

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if await self.answer_eta_question_if_needed(message) or message.content.endswith("--eta"):
            return

        if not await self.should_process_message(message):
            return

        self.rate_limiter.register_request(message.author.id)
        await self.ai_bot.memorize_short_term(message)
        await self.vector_db_conn.add_messages([message])
        reply = await message.reply(f"{KamiChan.Vocabulary.EMOJI_UWU} {BOT_NAME} is typing...")
        try:
            disclaimer = f"{KamiChan.Vocabulary.EMOJIS_COMBO_UNOFFICIAL} | [Learn more.](https://discord.com/channels/532557135167619093/1192649325709381673/1196285641978302544)"
            resp = DiscordBotResponse(self.ai_bot)
            resp_str = await resp.create(message)
            reply_msg = await reply.edit(content=resp_str)
            await self.ai_bot.memorize_short_term(reply_msg)
            await self.vector_db_conn.add_messages([reply_msg])
            if resp.verbose:
                log_file = io.StringIO(resp.verbose_log)
                await reply.edit(content=resp_str + "\n" + disclaimer, attachments=[discord.File(log_file, filename="verbose_log.txt")])
            else:
                await reply.edit(content=resp_str + "\n" + disclaimer)
        except Exception as e:
            await self.ai_bot.forget_short_term(message)
            await self.ai_bot.forget_short_term(reply)
            traceback.print_exc()
            await reply.edit(content=f"Sorry, there was an error!! {KamiChan.Vocabulary.EMOJI_DESPAIR} ```{str(e)}```")