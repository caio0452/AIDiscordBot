from functools import wraps
from ai_apis import providers
from ai_apis.client import LLMClient
from ai_apis.types import LLMRequestParams, Prompt
from bot_workflow.profile_loader import Profile
from bot_workflow.knowledge import KnowledgeIndex, LongTermMemoryIndex
from bot_workflow.types import AIBotData, MemorizedMessage, MemorizedMessageHistory, SynchronizedMessageHistory

import re
import discord
import datetime
import traceback

class CustomBotData(AIBotData):
    def __init__(self,
                 *,
                 name: str,
                 profile: Profile,
                 provider_store: providers.ProviderDataStore,
                 knowledge: KnowledgeIndex,
                 long_term_memory: LongTermMemoryIndex,
                 discord_bot_id: int,
                 memory_length: int
                ):
        super().__init__(name, MemorizedMessageHistory(memory_length=memory_length))
        self.profile = profile
        self.provider_store = provider_store
        self.discord_bot_id = discord_bot_id
        self.long_term_memory = long_term_memory # TODO: unused
        self.recent_history = SynchronizedMessageHistory()
        self.knowledge = knowledge 
        self.RECENT_MEMORY_LENGTH = profile.recent_message_history_length

class ResponseLogger:
    def __init__(self):
        self.text = ""

    def verbose(self, text: str, *, category: str | None = None):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if category:
            self.text += f"\n[{current_time}] --- {category} ---\n{text}\n"
        else:
            self.text += f"\n[{current_time}] {text}\n"

def response_step(NAME):
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            try:
                params = self.bot_data.profile.request_params[NAME]
                self.logger.verbose(f"Initiating step {NAME}. {params}", category=NAME)

                result = await func(self, *args, **kwargs)

                self.logger.verbose(f"Step {NAME} done, {result}", category=NAME)
                return result
            except Exception as e:
                self.logger.verbose(f"Error in {NAME}: {e}", category="ERROR")
                traceback.print_exc()
                raise e 
        return wrapper
    return decorator

class DiscordBotResponse:
    def __init__(self, bot_data: CustomBotData, verbose: bool=False):
        self.verbose = verbose
        self.bot_data = bot_data
        self.logger: ResponseLogger = ResponseLogger()
        self.clients: dict[str, LLMClient] = {}

        for k, v in bot_data.provider_store.providers.items():
            self.clients[k] = LLMClient.from_provider(v)

    # TODO: clean this method up
    async def create(self, message: discord.Message) -> str:
        MAIN_CLIENT_NAME = "PERSONALITY"
        USABLE_HISTORY_LENGTH = 14
        FALLBACKS = ["llama-3-8b"] # TODO: don't hardcode this
        usable_history = await self.bot_data.recent_history.get_finalized_message_history()
        last_n_messages = [msg for msg in usable_history._memory][-USABLE_HISTORY_LENGTH:]
        last_n_messages.append(await MemorizedMessage.of_discord_message(message))
        usable_messages = MemorizedMessageHistory(last_n_messages)
        full_prompt = await self.build_full_prompt(
            usable_messages, 
            message
        )
        default_params = self.bot_data.profile.request_params[MAIN_CLIENT_NAME]
        model_names_order = [default_params.model_name]
        model_names_order.extend(FALLBACKS)

        for name in model_names_order:
            try:
                current_params: LLMRequestParams = default_params.model_copy()
                current_params.model_name = name
                response = await self.clients[MAIN_CLIENT_NAME].send_request(
                    prompt=full_prompt,
                    params=current_params
                )
                personality_rewrite = await self.personality_rewrite(response.message.content)
                answer_with_replacements = personality_rewrite
                for k, v in self.bot_data.profile.regex_replacements.items():
                    answer_with_replacements = re.sub(k, v, answer_with_replacements)
                return answer_with_replacements
            except Exception as e:
                traceback.print_exc()
                self.logger.verbose(f"Request to LLM '{name}' failed with error: {e}", category="MODEL FAILURE")
        
        raise RuntimeError("Could not generate response and all fallbacks failed")
    
    @response_step("PERSONALITY_REWRITE")
    async def personality_rewrite(self, message: str) -> str:
        NAME = "PERSONALITY_REWRITE"
        name_prompt = self.bot_data.profile.prompts[NAME]
        prompt = name_prompt.replace({
            "message": message
        })
        response = await self.send_llm_request(
            name=NAME,
            prompt=prompt
        ) 
        self.logger.verbose(f"PROMPT: {prompt}", category="PERSONALITY REWRITE") 
        return response.message.content

    @response_step("USER_QUERY_REPHRASE")
    async def user_query_rephrase(self) -> str:
        NAME = "USER_QUERY_REPHRASE"
        recent_history_list = self.bot_data.recent_history.backing_history.as_list()
        user_prompt_str = "\n".join(
            [memorized_message.text for memorized_message in recent_history_list]
        )
        last_user = recent_history_list[-1].nick
        prompt = self.bot_data.profile.prompts[NAME].replace({
            "user_query": user_prompt_str, 
            "last_user": last_user
        })
        response = await self.send_llm_request(
            name=NAME,
            prompt=prompt
        )
        self.logger.verbose(f"Prompt: {prompt}\nResponse: {response}", category=NAME)
        return response.message.content

    @response_step("INFO_SELECT")
    async def info_select(self, user_query: str) -> str | None:
        NAME = "INFO_SELECT"
        user_prompt_str = ""
        knowledge_list = await self.bot_data.knowledge.retrieve(user_query)

        if len(knowledge_list) == 0:
            return None

        for knowledge in knowledge_list:
            text_content: str = knowledge["text"]
            user_prompt_str += f"INFO:n{text_content}"

        user_prompt_str += "QUERY: " + user_query
        prompt = self.bot_data.profile.prompts[NAME] \
            .replace({
                "user_query": user_prompt_str
            })
        response = await self.send_llm_request(
            name=NAME,
            prompt=prompt
        )
        self.logger.verbose(f"Prompt: {prompt}\nResponse: {response}", category=NAME)
        return response.message.content

    @response_step("PERSONALITY")
    async def build_full_prompt(self, memory_snapshot: MemorizedMessageHistory, original_msg: discord.Message) -> Prompt:
        NAME = "PERSONALITY"
        now_str = datetime.datetime.now().strftime("%B %d, %H:%M:%S")
        user_query = await self.user_query_rephrase()
        knowledge = await self.info_select(user_query)
        old_memories: str = "" # TODO: implement
        full_prompt: Prompt = self.bot_data.profile.prompts[NAME].model_copy()

        if knowledge is not None:
            knowledge_str = f"\n[INFO FROM KNOWLEDGE DB]:\n{knowledge}\n"
            self.logger.verbose(knowledge, category="INFO FROM KNOWLEDGE DB")
        else:
            knowledge_str = ""
            self.logger.verbose("The knowledge database has nothing relevant", category="INFO FROM KNOWLEDGE DB")

        for memorized_message in memory_snapshot.as_list():
            if memorized_message.is_bot:
                full_prompt.append(Prompt.assistant_msg(memorized_message.text))
            else:
                full_prompt.append(Prompt.user_msg(memorized_message.text))

        img_desc = await self.describe_image_if_present(original_msg)
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

    # TODO: clean this method up
    async def describe_image_if_present(self, message) -> str | None:
        NAME = "IMAGE_VIEW"
        if len(message.attachments) == 1:
            if message.channel.nsfw:
                await message.reply(":X: I can't see attachments in NSFW channels!")
                return None
            attachment = message.attachments[0]
            if attachment.content_type.startswith("image/"):
                await message.add_reaction("👀")
                # Todo: only last message is possibly not enough context
                response = await self.clients[NAME].send_request(
                    prompt=Prompt(
                                messages=[
                                    Prompt.user_msg(
                                        content=f"Describe the image in a sufficient way to answer the following query: '{message.content}'" \
                                        "If the query is empty, just describe the image. ",
                                        image_url=attachment.url
                                    )
                                ]
                            ),
                   params=LLMRequestParams(
                         model_name="openai/gpt-4o"
                    )
                )
                return response.message.content


    async def send_llm_request(self, *, name: str, prompt: Prompt):
        params = self.bot_data.profile.request_params[name]
        provider: providers.ProviderData = self.bot_data.profile.providers[name]
        client: LLMClient = LLMClient.from_provider(provider)

        return await client.send_request(prompt=prompt, params=params) 