from ai import OAICompatibleProvider
from abc import ABC, abstractmethod
from abc import abstractmethod
from typing import List
import numpy as np
import json

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
    def __init__(self, *, preset_question: str, embedding: List[float], required_matchers: List[PresetQueryMatcher], min_required_similarity: float, answer:str):
        self.preset_question = preset_question
        self.embedding = embedding
        self.required_matchers = required_matchers
        self.min_required_similarity = 0.5
        self.answer = answer

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
    
_cached_manager = None
async def manager(embeddings_client: OAICompatibleProvider) -> PresetQueryManager:
    json_file_path = "queries.json"
    global _cached_manager
    if _cached_manager is not None:
        return _cached_manager

    print("Building query db...")
    presets_manager = PresetQueryManager(embeddings_client)

    with open(json_file_path, 'r') as file:
        data = json.load(file)
        queries = data["queries"]

    for query in queries:
        question = query["question"]
        keywords = query["keywords"]
        min_similarity = query["min_similarity"]
        answer = query["answer"]

        embedding = await embeddings_client.vectorize(question)
        await presets_manager.add_query(
            PresetQuery(
                preset_question=question,
                embedding=embedding,
                required_matchers=[
                    KeywordMatcher(keywords),
                    EmbeddingSimilarityMatcher(embedding, min_similarity)
                ],
                min_required_similarity=min_similarity,
                answer=answer
            )
        )

    print("Done")
    _cached_manager = presets_manager
    return presets_manager