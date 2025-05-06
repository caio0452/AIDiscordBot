import os
import hashlib
import numpy as np

from typing import Any
from txtai import Embeddings
from dataclasses import dataclass
from core.ai_apis.providers import ProviderData
from core.ai_apis.client import SyncEmbeddingsClient

class VectorDatabase:
    MEMORIES_PATH = "../memories/"

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
        SUBINDEXES = ["knowledge", "memories"]
        self.vectorizer = SyncEmbeddingsClient(provider)

        def external_transform(inputs): 
            # TODO: this is sync. Is that a concern?
            resp = self.vectorizer.vectorize(input=inputs)
            return np.array(resp, dtype=np.float32)

        self._db_data: dict[str, Embeddings]
        memory_files = [
            file for file in os.listdir(VectorDatabase.MEMORIES_PATH) 
            if file.endswith('.tar.gz')
        ]

        for file in memory_files:
            index_name = file.removesuffix(".tar.gz")
            self._db_data[index_name] = Embeddings(
                config={
                    "transform": external_transform,
                    "backend": "faiss", 
                    "content": True
                }
            )
            self._db_data[index_name].load(VectorDatabase.MEMORIES_PATH + file)

        for subindex in SUBINDEXES:
            already_loaded_from_file = subindex in self._db_data
            if already_loaded_from_file:
                continue
            self._db_data[subindex] = Embeddings(
                config={
                    "transform": external_transform,
                    "backend": "faiss", 
                    "content": True
                }
            )
            self._db_data[subindex].initindex(False)

    def all_indexes(self) -> list[Embeddings]:
        return list(self._db_data.values())
    
    def get_index(self, index_name: str) -> Embeddings:
        return self._db_data[index_name]
        
    def search(self, data: str, limit: int=5, index_name: str | None = None) -> list[Any]:
        ret: list[Any] = []
        if index_name is None:
            for index in self.all_indexes():
                ret.append(index.search(data, limit=limit))
        else:
            target_index = self.get_index(index_name)
            return target_index.search(data)
        if not isinstance(ret, list):
            raise ValueError("Search returned unknown non-list item: ", ret)
        return ret

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
        for index_name, index in self._db_data.items():
            index.save(f"{VectorDatabase.MEMORIES_PATH}{index_name}.tar.gz")