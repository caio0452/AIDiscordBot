import re
import os
import glob
import asyncio

from ai_apis.providers import ProviderData
from bot_workflow.vector_db import VectorDatabase
from bot_workflow.memorized_message import MemorizedMessage

class LongTermMemoryIndex:
    def __init__(self, provider: ProviderData): 
        self.vector_db: VectorDatabase = VectorDatabase(provider)

    def memorize(self, message: MemorizedMessage):
        self.vector_db.index(VectorDatabase.Entry(message.text, str(message), message.message_id))

    async def get_closest_messages(self, reference: str, *, n=5) -> list:
        return await self.vector_db.search(reference, n)

class KnowledgeIndex:
    def __init__(self, provider): 
        self.vector_db = VectorDatabase(provider)

    @staticmethod
    def chunk_text(text, chunk_size=2000, overlap=400):
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

    async def index_text(self, text, *, metadata="knowledge"):
        for chunk in KnowledgeIndex.chunk_text(text):
            self.vector_db.index(VectorDatabase.Entry(
                data=chunk, 
                metadata=metadata,
                entry_id=None
            ))

    async def index_texts(self, texts: list[str], *, metadata="knowledge"):
        entries = [VectorDatabase.Entry(text, "knowledge", None) for text in texts]
        await self.vector_db.mass_index(entries)

    async def index_from_folder(self, path, max_concurrent_tasks=8): 
        if not os.path.exists(path):
            print(f"The knowledge folder, located in '{path}' does not exist. Skipping knowledge indexing.")
            return

        all_files = glob.glob(f"{path}/*")
        txt_files = [file for file in all_files if file.endswith('.txt')]
        non_txt_files = [file for file in all_files if not file.endswith('.txt')]

        for file in non_txt_files:
            print(f"Error: {file} is not a .txt file. All knowledge must be in text files. Skipping.")

        if not txt_files:
            print(f"No files in knowledge folder: '{path}', nothing to index'")
            return

        async def process_file(file_path):
            with open(file_path, 'r') as file:
                text = file.read()
                chunks = KnowledgeIndex.chunk_text(text)
                await self.index_texts(chunks)
                return len(chunks)

        tasks = [process_file(file) for file in txt_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        total_chunks = 0
        for file_path, result in zip(txt_files, results):
            if isinstance(result, Exception):
                print(f"Error indexing {file_path}: {result}")
            elif isinstance(result, int):
                total_chunks += result
                print(f"Indexed {file_path}: {result} chunks")

        print(f"Total chunks indexed: {total_chunks}")

    async def retrieve(self, related_text):
        return await self.vector_db.search(related_text)