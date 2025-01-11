from abc import ABC
import asyncio
from bot_workflow.memorized_message import MemorizedMessage

class MemorizedMessageHistory:
    def __init__(self, initial_history: list[MemorizedMessage]  | None = None, memory_length: int = 14):
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

    async def add_after(self, id: int, new_message: MemorizedMessage) -> bool:
        for index, msg in enumerate(self._memory):
            if msg.message_id == id:
                self._memory.insert(index + 1, new_message)
                if len(self._memory) > self.MEMORY_LENGTH:
                    self._memory.pop(0)
                return True
        return False 
    
    async def remove(self, message: MemorizedMessage):
        self._memory = [mem_msg for mem_msg in self._memory if mem_msg.message_id != message.message_id]

    def clone(self):
        new_instance = MemorizedMessageHistory(
            initial_history=[m for m in self._memory], 
            memory_length=self.MEMORY_LENGTH
        )
        return new_instance

    def as_list(self) -> list[MemorizedMessage]:
        return self._memory
    
    def __str__(self) -> str:
        ret = ""
        for msg in self._memory:
            id = msg.message_id
            bot_str = "" if not msg.is_bot else "(BOT)"
            ret += f"[ID {id} | {msg.sent}] <{msg.nick}{bot_str}> {msg.text}"
        return ret

class SynchronizedMessageHistory:
    def __init__(self, history: MemorizedMessageHistory = MemorizedMessageHistory()):
        self.backing_history = history
        self._pending_message_ids: list[int] = []
        self._lock = asyncio.Lock()

    async def add(self, message: MemorizedMessage, *, pending=False):
        async with self._lock:
            await self.backing_history.add(message)
            if pending:
                self._pending_message_ids.append(message.message_id)
        
    async def mark_finalized(self, message_id: int) -> bool:
        async with self._lock:
            if message_id in self._pending_message_ids:
                self._pending_message_ids.remove(message_id)
                return True
            else:
                return False
        
    def is_pending(self, message_id: int) -> bool:
        return message_id in self._pending_message_ids

    async def get_finalized_message_history(self) -> MemorizedMessageHistory:
        ret_msgs = []

        async with self._lock:
            for msg in self.backing_history._memory:
                if not self.is_pending(msg.message_id):
                    ret_msgs.append(msg)
            return MemorizedMessageHistory(ret_msgs)
        
    def __str__(self) -> str:
        ret = ""
        for msg in self.backing_history._memory:
            id = msg.message_id
            pending_str = "" if not self.is_pending(id) else "(PENDING)"
            bot_str = "" if not msg.is_bot else "(BOT)"
            ret += f"{pending_str}[ID {id} | {msg.sent}] <{msg.nick}{bot_str}> {msg.text}"
        return ret
        
class AIBotData(ABC):
    def __init__(self, name: str, recent_memory: MemorizedMessageHistory | None = None):
        self.name = name
        self.memory = recent_memory
