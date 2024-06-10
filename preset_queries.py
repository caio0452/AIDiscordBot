from ai import OAICompatibleProvider
from abc import ABC, abstractmethod
from abc import abstractmethod
from typing import List
import numpy as np

class UserUtterance:
    def __init__(self, query: str, embedding: List[float]):
        self.query = query
        self.embedding = embedding

class PresetQueryMatcher(ABC):
    @abstractmethod
    def matches_utterance(self, utterance: UserUtterance) -> bool:
        raise NotImplementedError()

class KeywordMatcher(PresetQueryMatcher):
    def __init__(self, keywords: List[str], ignore_case: bool = True):
        self.keywords = keywords
        self.ignore_case = ignore_case

    def matches_utterance(self, utterance: UserUtterance) -> bool:
        if self.ignore_case:
            return any(keyword.lower() in utterance.query.lower() for keyword in self.keywords)
        else:
            return any(keyword in utterance.query for keyword in self.keywords)

class EmbeddingSimilarityMatcher(PresetQueryMatcher):
    def __init__(self, embedding: List[float], min_thresh: float):
        self.embedding = embedding  # TODO: assumes normalized vector
        self.min_thresh = min_thresh

    def matches_utterance(self, utterance: UserUtterance) -> bool:
        return np.dot(self.embedding, utterance.embedding) >= self.min_thresh

class PresetQuery:
    def __init__(self, *, preset_question: str, embedding: List[float], required_matchers: List[PresetQueryMatcher]):
        self.preset_question = preset_question
        self.embedding = embedding
        self.required_matchers = required_matchers

    def matches(self, utterance: UserUtterance) -> bool:
        return all(matcher.matches_utterance(utterance) for matcher in self.required_matchers)

class PresetQueryManager:
    def __init__(self, embeddings_client: OAICompatibleProvider):
        self.embeddings_client = embeddings_client
        self._all_queries: List[PresetQuery] = []

    async def add_query(self, query: PresetQuery):
        self._all_queries.append(query)

    async def get_all_matching_user_utterance(self, utterance: str) -> List[PresetQuery]:
        ret = []
        utterance_emb = await self.embeddings_client.vectorize(utterance)
        user_utterance = UserUtterance(query=utterance, embedding=utterance_emb)
        for query in self._all_queries:
            if query.matches(user_utterance):
                ret.append(query)
        return ret

    @property
    def all_preset_queries(self):
        return self._all_queries
    
