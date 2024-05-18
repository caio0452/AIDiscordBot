import discord
import openai
from datetime import datetime
from typing import List
from memorized_message import MemorizedMessage
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dataclasses import dataclass
from ai import OAICompatibleProvider

@dataclass
class VectorSearchResult:
    vector: list[float]
    payload: str
    score: float

class QdrantVectorDbConnection:
    def __init__(self, qdrant_client: QdrantClient, openai_client: openai.AsyncOpenAI, vector_dimension: int):
        self.qdrant_client = qdrant_client
        self.openai_client = OAICompatibleProvider(openai_client)
        self.vector_dimension = vector_dimension
        self.upserted_count = 0

        collection_names = [col.name for col in self.qdrant_client.get_collections().collections]

        # TODO: check if files changed instead, we don't always need to re-index
        if "messages" in collection_names:
            self.qdrant_client.delete_collection("messages")
        if "knowledge" in collection_names:
            self.qdrant_client.delete_collection("knowledge")
        if "qa_knowledge" in collection_names:
            self.qdrant_client.delete_collection("qa_knowledge")

        self.qdrant_client.create_collection(
            collection_name="messages",
            vectors_config=models.VectorParams(size=3072, distance=models.Distance.DOT),
        )
        self.qdrant_client.create_collection(
            collection_name="knowledge",
            vectors_config=models.VectorParams(size=3072, distance=models.Distance.DOT),
        )
        self.qdrant_client.create_collection(
            collection_name="qa_knowledge",
            vectors_config=models.VectorParams(size=3072, distance=models.Distance.DOT),
        )

    def _collection_exists(self, name: str) -> bool:
        try:
            self.qdrant_client.get_collection(name)
            return True
        except:
            return False

    async def add_messages(self, messages: List[discord.Message]):
        payloads = []
        message_contents = [message.content for message in messages]
        vectors = await self.openai_client.vectorize_many(message_contents)

        for i, message in enumerate(messages):
            payload = {
                "text": message.content,
                "nick": message.author.nick or message.author.name,
                "sent": message.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "is_bot": message.author.bot,
            }
            payloads.append({
                "id": self.upserted_count + i,
                "vector": vectors[i],
                "payload": payload
            })
        self.qdrant_client.upsert(
            collection_name="messages",
            points=payloads
        )
        self.upserted_count += len(payloads)

    async def add_qa_knowledge(self, qa_pairs: List[dict[str, str]]):
        questions = [qa_pair["question"] for qa_pair in qa_pairs]
        vectors = await self.openai_client.vectorize_many(questions)

        payloads = []
        for i, qa_pair in enumerate(qa_pairs):
            payload = {"text": qa_pair["answer"]}
            payloads.append({
                "id": self.upserted_count + i,
                "vector": vectors[i],
                "payload": payload
            })
        self.qdrant_client.upsert(
            collection_name="qa_knowledge",
            points=payloads
        )

        self.upserted_count += len(payloads)

    async def add_text_knowledge(self, knowledge_list: List[str]):
        vectors = await self.openai_client.vectorize_many(knowledge_list)
        payloads = []

        for i, paragraph in enumerate(knowledge_list):
            payload = {"text": paragraph}
            payloads.append({
                "id": self.upserted_count + i,
                "vector": vectors[i],
                "payload": payload
            })

        self.qdrant_client.upsert(
            collection_name="knowledge",
            points=payloads
        )

        self.upserted_count += len(payloads)

    async def query_relevant_messages(self, query: str) -> List[MemorizedMessage]:
        vector = await self.openai_client.vectorize(query)
        search_results = self.qdrant_client.search(
            collection_name="messages",
            query_vector=vector,
            query_filter=None,
            limit=5,
            with_payload=True,
            search_params=models.SearchParams(hnsw_ef=128, exact=True),
        )
        IRRELEVANT_MSG_ID = -1
        msgs = [
            MemorizedMessage(text=hit.payload['text'], nick=hit.payload['nick'],
                             sent=datetime.strptime(hit.payload['sent'], "%Y-%m-%d %H:%M:%S"),
                             is_bot=hit.payload['is_bot'],
                             message_id=IRRELEVANT_MSG_ID)
            for hit in search_results
        ]
        return msgs

    async def query_relevant_knowledge(self, query: str) -> List[VectorSearchResult]:
        vector = await self.openai_client.vectorize(query)
        search_results = self.qdrant_client.search(
            collection_name="knowledge",
            query_vector=vector,
            query_filter=None,
            limit=5,
            with_payload=True,
            score_threshold=0.3
        )

        ret: list[VectorSearchResult] = []
        for result in search_results:
            ret.append(VectorSearchResult(result.vector, result.payload['text'], result.score))

        return ret

    async def query_qa_knowledge(self, question: str) -> List[VectorSearchResult]:
        vector = await self.openai_client.vectorize(query)

        search_results = self.qdrant_client.search(
            collection_name="qa_knowledge",
            query_vector=vector,
            query_filter=None,
            limit=5,
            with_payload=True,
        )

        ret: List[VectorSearchResult] = []
        for result in search_results:
            ret.append(VectorSearchResult(result.vector, result.payload['text'], result.score))

        return ret

class QdrantVectorDb:
    def __init__(self, host: str, openai_client: openai.AsyncOpenAI, port: int, vector_dimension: int):
        self.host = host
        self.port = port
        self.openai_client = openai_client
        self.vector_dimension = vector_dimension

    def connect(self) -> QdrantVectorDbConnection:
        client = QdrantClient(host=self.host, port=self.port)
        return QdrantVectorDbConnection(client, self.openai_client, self.vector_dimension)
