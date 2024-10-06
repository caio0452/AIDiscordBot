from abc import ABC, abstractmethod
from typing import List
from memorized_message import MemorizedMessage

class BotMemory:
    def __init__(self, initial_memories: list[MemorizedMessage] = [], memory_length: int = 5):
        self._memory = initial_memories
        self.MEMORY_LENGTH = memory_length

    async def memorize_short_term(self, message: MemorizedMessage, sanitize_msg):
        self._memory.append(message)
        if len(self._memory) > self.MEMORY_LENGTH:
            self._memory.pop(0)

    async def forget_short_term(self, message: MemorizedMessage):
        self._memory = [mem_msg for mem_msg in self._memory if mem_msg.message_id != message.message_id]

    def clone(self):
        new_instance = BotMemory(
            initial_memories=[m for m in self._memory], 
            memory_length=self.MEMORY_LENGTH
        )
        return new_instance

    def without_dupe_ending_user_msgs(self) -> "BotMemory":
        ret: list[MemorizedMessage] = []

        for i, mem in enumerate(self._memory):
            ret.append(mem)
            if i < len(self._memory) - 1:
                duplicate_user_messages = (not self._memory[i].is_bot) and (not self._memory[i + 1].is_bot)
                if duplicate_user_messages:
                    # Only keep last user message
                    ret.pop(-1)
                    ret.append(self._memory[-1])
                    return BotMemory(initial_memories=ret)

        return BotMemory(initial_memories=ret)

    def get_memory(self) -> list[MemorizedMessage]:
        return self._memory

class AIBotData(ABC):
    def __init__(self, name: str, memory: BotMemory | None = None):
        self.name = name
        self.memory = memory

