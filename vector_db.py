import hashlib
import numpy as np

from typing import Any
from openai import AsyncOpenAI
from txtai import Embeddings
from providers import Provider
from ai import EmbeddingsClient

class VectorDatabase:
    def __init__(self, provider: Provider): 
        self.vectorizer = EmbeddingsClient(provider)

        def transform(inputs): # TODO: this is sync. Performance concern?
            resp = self.vectorizer.vectorize(input=inputs)
            print(f"Response: {resp}")
            return np.array(resp, dtype=np.float32)

        self.db_data = Embeddings(
            config={
                "transform": transform, 
                "backend": "numpy", 
                "content": True,
                "indexes": {
                    "knowledge": {},
                    "messages": {}
                }
            }
        )

        self.db_data.initindex(False)

    async def search(self, data: str, limit: int=5, index_name: str | None = None) -> list[Any]:
        if index_name is None:
            ret = self.db_data.search(data, limit=limit)
        else:
            ret = self.db_data.search(data, limit=limit, index=index_name)

        if not isinstance(ret, list):
            raise RuntimeError(f"Expected database search to return list, not object {str(ret)} of type {type(ret)}")
        return ret

    async def get_index(self, index_name: str):
        indexes = self.db_data.indexes
        if indexes is None:
            raise RuntimeError("There are no indexes to search")
        try:
            return indexes[index_name]
        except Exception as e:
            raise RuntimeError(f"Failed to access index {index_name}") from e

    async def index(self, *, index_name: str, data: str, metadata: str, entry_id: int | None):
        target_index = await self.get_index(index_name)

        if entry_id is None:
            combined = data + metadata
            id = int(hashlib.sha256(combined.encode()).hexdigest(), 16) & 0xFFFFFFFF  
        else:
            id = entry_id

        target_index.index((id, data, metadata))

    async def delete_ids(self, *, index_name: str, entry_ids: list[int]) -> int:
        target_index = await self.get_index(index_name)
        return target_index.remove_ids(entry_ids)