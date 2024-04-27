from abc import ABC, abstractmethod
from typing import List
from memorized_message import MemorizedMessage
import discord

class AIBot(ABC):
    def __init__(self, name: str, memory: List[MemorizedMessage] | None = None):
        self.name = name
        self.memory = memory

    @abstractmethod
    async def respond_to_query(self, message: discord.Message) -> str:
        raise NotImplementedError()

