from abc import ABC
from bot_workflow.memorized_message import MemorizedMessage

class MemorizedMessageHistory:
    def __init__(self, initial_history: list[MemorizedMessage]  | None = None, memory_length: int = 8):
        if initial_history is None:
            self._memory = []
        else:
            self._memory = [msg for msg in initial_history]
            self._memory = self._memory[-memory_length:]
        self.MEMORY_LENGTH = memory_length

    async def add(self, message: MemorizedMessage):
        self._memory.append(message)
        if len(self._memory) > self.MEMORY_LENGTH:
            self._memory.pop(0)

    async def remove(self, message: MemorizedMessage):
        self._memory = [mem_msg for mem_msg in self._memory if mem_msg.message_id != message.message_id]

    def clone(self):
        new_instance = MemorizedMessageHistory(
            initial_history=[m for m in self._memory], 
            memory_length=self.MEMORY_LENGTH
        )
        return new_instance

    def without_dupe_ending_user_msgs(self) -> "MemorizedMessageHistory":
        ret: list[MemorizedMessage] = []

        for i, mem in enumerate(self._memory):
            ret.append(mem)
            if i < len(self._memory) - 1:
                duplicate_user_messages = (not self._memory[i].is_bot) and (not self._memory[i + 1].is_bot)
                if duplicate_user_messages:
                    # Only keep last user message
                    ret.pop(-1)
                    ret.append(self._memory[-1])
                    return MemorizedMessageHistory(initial_history=ret)

        return MemorizedMessageHistory(initial_history=ret)

    def as_list(self) -> list[MemorizedMessage]:
        return self._memory

class AIBotData(ABC):
    def __init__(self, name: str, recent_memory: MemorizedMessageHistory | None = None):
        self.name = name
        self.memory = recent_memory
