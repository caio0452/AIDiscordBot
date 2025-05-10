from core.ai_apis.client import LLMClient
from core.ai_apis.types import LLMRequestParams, Prompt
from core.bot_workflow.response_logs import ResponseLogger
from core.bot_workflow.custom_bot_data import CustomBotData
from core.bot_workflow.types import MemorizedMessage, MemorizedMessageHistory
from core.bot_workflow.response_steps import PersonalityRewriteStep, RelevantInfoSelectStep, UserQueryRephraseStep

import re
import discord
import datetime
import traceback

class DiscordBotResponse:
    def __init__(self, bot_data: CustomBotData, verbose: bool=False):
        self.verbose = verbose
        self.bot_data = bot_data
        self.logger: ResponseLogger = ResponseLogger()
        self.clients: dict[str, LLMClient] = {}

        for provider_name, provider_data in bot_data.provider_store.providers.items():
            self.clients[provider_name] = LLMClient.from_provider(provider_data)

    async def create(self, message: discord.Message) -> str:
        full_prompt = await self._build_full_prompt(message)
        MAIN_CLIENT_NAME = "PERSONALITY"
        default_params = self.bot_data.profile.request_params[MAIN_CLIENT_NAME]
        model_names_order = [default_params.model_name] + self.bot_data.profile.options.llm_fallbacks
        exc_details = ""

        for name in model_names_order:
            try:
               modified_params = default_params.model_copy(deep=True)
               modified_params = LLMRequestParams(
                   model_name=name,
                   temperature=default_params.temperature,
                   max_tokens=default_params.max_tokens,
                   logit_bias=default_params.logit_bias
               )
               return await self._respond_in_character(
                   prompt=full_prompt, 
                   params=modified_params,
                   message=message.content
                )
            except Exception as e:
                traceback.print_exc()
                exc_details += traceback.format_exc()
                self.logger.verbose(f"Request to LLM '{name}' failed with error: {e}", category="MODEL FAILURE")
        
        raise RuntimeError("Could not generate response and all fallbacks failed")
    
    async def _respond_in_character(self, *, prompt: Prompt, params: LLMRequestParams, message: str):
        MAIN_CLIENT_NAME = "PERSONALITY"
        response = await self.clients[MAIN_CLIENT_NAME].send_request(
            prompt=prompt,
            params=params
        )
        self.logger.verbose(response.message.content, category="PRE-REWRITE MESSAGE")
        personality_rewriter = PersonalityRewriteStep()
        personality_rewrite = await personality_rewriter.execute(self.bot_data, message) 
        if personality_rewrite is None:
            raise RuntimeError("Personality rewrite step returned empty response")
        for target, replacement in self.bot_data.profile.regex_replacements.items():
            personality_rewrite = re.sub(target, replacement, personality_rewrite)
        return personality_rewrite
    
    async def _get_usable_message_history_before(self, message: discord.Message) -> MemorizedMessageHistory:
        USABLE_HISTORY_LENGTH = 14
        usable_history = await self.bot_data.recent_history.get_finalized_message_history()
        last_n_messages = [msg for msg in usable_history._memory][-USABLE_HISTORY_LENGTH:]
        last_n_messages.append(await MemorizedMessage.of_discord_message(message))
        return MemorizedMessageHistory(last_n_messages)

    async def _build_full_prompt(self, original_msg: discord.Message) -> Prompt:
        NAME = "PERSONALITY"
        memory_snapshot = await self._get_usable_message_history_before(original_msg)
        now_str = datetime.datetime.now().strftime("%B %d, %H:%M:%S")
        old_memories: str = "" 
        knowledge = None
        knowledge_str = ""
        user_query = original_msg.content
        full_prompt: Prompt = self.bot_data.profile.get_prompt(NAME)

        for memorized_message in memory_snapshot.as_list():
            if memorized_message.is_bot:
                full_prompt.append(Prompt.assistant_msg(memorized_message.text))
            else:
                full_prompt.append(Prompt.user_msg(memorized_message.text))

        if self.bot_data.profile.options.enable_knowledge_retrieval:
            rephrase = await UserQueryRephraseStep().execute(self.bot_data, original_msg.content)
            if rephrase is None:
                raise RuntimeError("Rephraser step returned empty response")
            
            info_selector = RelevantInfoSelectStep(user_query=rephrase)
            knowledge = await info_selector.execute(self.bot_data, original_msg.content)
            if knowledge is None:
                raise RuntimeError("Knowledge retrieval step returned empty response")
            
            if self.bot_data.long_term_memory is not None:
                for msg in await self.bot_data.long_term_memory.get_closest_messages(rephrase):
                    old_memories += str(msg) # TODO: establish a type for this

            user_query = rephrase
            knowledge_str = f"\n[INFO FROM KNOWLEDGE DB]:\n{knowledge}\n"
            self.logger.verbose(knowledge, category="INFO FROM KNOWLEDGE DB")
        
        img_desc = await self._describe_image_if_present(original_msg, user_query)
        if img_desc:
            full_prompt.append(Prompt.user_msg(img_desc))

        full_prompt = full_prompt.replace({
            "now": now_str,
            "nick": original_msg.author.display_name,
            "knowledge": knowledge_str,
            "old_memories": old_memories
        })

        self.logger.verbose(f"FULL PROMPT: {full_prompt}", category="FULL PROMPT")
        self.logger.verbose(str(self.bot_data.recent_history), category="FULL MEMORY DUMP")
        self.logger.verbose(str(memory_snapshot), category="USABLE MEMORY DUMP")
        return full_prompt

    async def _describe_image_if_present(self, message: discord.Message, user_query: str) -> str | None:
        NAME = "IMAGE_VIEW"

        if len(message.attachments) == 0:
            return None
        if len(message.attachments) > 1:
            for emoji in ["❌", "1️⃣", "🖼️"]:
                await message.add_reaction(emoji)
            return None
        if isinstance(message.channel, discord.TextChannel) and message.channel.nsfw:
            await message.reply(":x: I can't see attachments in NSFW channels!")

        attachment = message.attachments[0]
        if not (attachment.content_type and attachment.content_type.startswith("image/")):
            return None
        
        await message.add_reaction("👀")
        response = await self.clients[NAME].send_request(
            prompt=Prompt(
                    messages=[
                        Prompt.user_msg(
                            content=f"Describe the image in detail, including a sufficient answer to the following query: '{message.content}'" \
                            "If the query is empty, just describe the image. ",
                            image_url=attachment.url
                        )
                    ]
                ),
            params=LLMRequestParams(model_name="openai/gpt-4o")
        )
        return response.message.content