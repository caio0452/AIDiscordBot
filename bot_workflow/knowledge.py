import re
import os
import glob

from bot_workflow.vector_db import VectorDatabase
from ai_apis.providers import ProviderData

# TODO: abstract this further
class LongTermMemoryIndex:
    def __init__(self, provider: ProviderData): 
        self.vector_db: VectorDatabase = VectorDatabase(provider)

class KnowledgeIndex:
    def __init__(self, provider: ProviderData): 
        self.vector_db: VectorDatabase = VectorDatabase(provider)

    @staticmethod
    def chunk_text(text: str, chunk_size=2000, overlap=400) -> list[str]:
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if start != 0:
                start -= overlap
            if end < len(text):
                # Don't cut off words
                boundary_end = re.search(r'\b', text[end:])
                if boundary_end is not None:
                    end = boundary_end.start() + end
            chunks.append(text[start:end])
            start += chunk_size
        return chunks

    async def index_from_folder(self, path: str):
        if not os.path.exists(path):
            print(f"The knowledge folder, located in '{path}' does not exist. Skipping knowledge indexing.")
            return

        for file_path in glob.glob(f"{path}/*"):
            if not file_path.endswith('.txt'):
                print(f"Error: {file_path} is not a .txt file, all knowledge must be in text files")
                continue

            with open(file_path, 'r') as file:
                await self.index_text(file.read())
                print(f"Indexed {file_path}")
                             
    async def index_text(self, text: str, *, metadata: str = "knowledge"):
        for chunk in KnowledgeIndex.chunk_text(text):
            await self.vector_db.index(
                data=chunk, 
                metadata=metadata,
                entry_id=None
            )

    async def retrieve(self, related_text: str):
        return await self.vector_db.search(related_text)