import hashlib
import numpy as np

from typing import Any
from txtai import Embeddings
from dataclasses import dataclass
from core.ai_apis.providers import ProviderData
from core.ai_apis.client import SyncEmbeddingsClient

class VectorDatabase:
    @dataclass
    class Entry:
        data: str
        metadata: dict[str, Any]
        entry_id: int | None = None

        def __post_init__(self):
            if self.entry_id is None:
                combined = self.data + str(self.metadata)
                self.entry_id = int(hashlib.sha256(combined.encode()).hexdigest(), 16) & 0x7FFFFFFF

        def as_txtai_object(self) -> tuple:
            return (self.entry_id, {"text": self.data, "metadata": self.metadata}, None)
        
    def __init__(self, provider: ProviderData):
        self.vectorizer = SyncEmbeddingsClient(provider)

        def external_transform(inputs): 
            # TODO: this is sync. Is that a concern?
            resp = self.vectorizer.vectorize(input=inputs)
            return np.array(resp, dtype=np.float32)

        self.db_data = Embeddings(
            config={
                "transform": external_transform,
                "backend": "faiss", 
                "content": True,
                "indexes": {
                    "knowledge": {},
                    "memories": {}
                }
            }
        )
        self.db_data.initindex(False)

    def search(self, data: str, limit: int=5, index_name: str | None = None) -> list[Any]:
        if index_name is None:
            ret = self.db_data.search(data, limit=limit)
        else:
            target_index = self.get_index(index_name)
            ret = target_index.search(data, limit=limit)
        if not isinstance(ret, list):
            raise ValueError("Search returned unknown non-list item: ", ret)
        return ret

    def get_index(self, index_name: str):
        indexes = self.db_data.indexes
        if indexes is None:
            raise RuntimeError("Vector database has no subindexes")
        
        try:
            return indexes.get(index_name)
        except KeyError:
             raise RuntimeError(f"Subindex '{index_name}' not found.")
        except Exception as e:
            raise RuntimeError(f"Error while accessign subindex: '{index_name}'") from e

    def index(self, index_name: str, entry: Entry):
        self.mass_index(index_name=index_name, entries=[entry])

    def mass_index(self, index_name: str, entries: list[Entry]):
        target_index = self.get_index(index_name)
        items_to_index = [entry.as_txtai_object() for entry in entries]
        
        try:
            target_index.index(items_to_index)
        except Exception as e:
            raise RuntimeError(f"Failed to index data into subindex '{index_name}'") from e

    def delete_ids(self, *, index_name: str, entry_ids: list[int]) -> int:
        target_index = self.get_index(index_name)
        try:
             deleted_ids = target_index.delete(entry_ids)
             return len(deleted_ids)
        except Exception as e:
            raise RuntimeError(f"Failed to delete ids {entry_ids} from subindex '{index_name}'") from e
        
    def save(self):
        self.db_data.save("../memories/memories.tar.gz")

    def load(self, path: str):
        self.db_data.load(path)