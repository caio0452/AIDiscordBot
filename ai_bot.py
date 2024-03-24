from abc import ABC, abstractmethod
from typing import List
from memorized_message import MemorizedMessage

class AIBot(ABC):
    def __init__(self, name: str, memory: List[MemorizedMessage] | None = None):
        self.name = name
        self.memory = memory

    @abstractmethod
    def respond_to_query(self, query: str) -> str:
        raise NotImplementedError()

