import hashlib
import numpy as np

from typing import Any
from txtai import Embeddings
from ai_apis.providers import ProviderData
from ai_apis.client import SyncEmbeddingsClient

class VectorDatabase:
    def __init__(self, provider: ProviderData): 
        self.vectorizer = SyncEmbeddingsClient(provider)

        def transform(inputs): # TODO: this is sync. Performance concern?
            print(f"Vectorizing this: {inputs}")
            resp = self.vectorizer.vectorize(input=inputs)
            return np.array(resp, dtype=np.float32)

        self.db_data = Embeddings(
            config={
                "transform": transform, 
                "backend": "numpy", 
                "content": True,
            }
        )

        self.db_data.index(["Hello, world"]) # Better way to init this?

    async def search(self, data: str, limit: int=5) -> list[Any]:
        ret = self.db_data.search(data, limit=limit)

        if not isinstance(ret, list):
            raise RuntimeError(f"Expected database search to return list, not object {str(ret)} of type {type(ret)}")
        print(f"Found: {ret}")
        return ret

    async def get_index(self, index_name: str):
        indexes = self.db_data.indexes
        if indexes is None:
            raise RuntimeError("There are no indexes to search")
        try:
            return indexes[index_name]
        except Exception as e:
            raise RuntimeError(f"Failed to access index {index_name}") from e

    async def index(self, *, data: str, metadata: str, entry_id: int | None):
        if entry_id is None:
            combined = data + metadata
            id = int(hashlib.sha256(combined.encode()).hexdigest(), 16) & 0xFFFFFFFF  
        else:
            id = entry_id

        self.db_data.upsert([(id, data, metadata)])

    async def delete_ids(self, *, index_name: str, entry_ids: list[int]) -> int:
        target_index = await self.get_index(index_name)
        return target_index.remove_ids(entry_ids)