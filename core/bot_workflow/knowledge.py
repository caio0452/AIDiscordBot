import re
import os
import glob
import asyncio
import hashlib

from core.ai_apis.providers import ProviderData
from core.bot_workflow.vector_db import VectorDatabase, VectorDatabaseConnection
from core.bot_workflow.memorized_message import MemorizedMessage

class LongTermMemoryIndex:
    def __init__(self, _db_conn: VectorDatabaseConnection): 
        self._db_conn = _db_conn

    @staticmethod
    async def from_provider(provider: ProviderData) -> "LongTermMemoryIndex":
        vector_db: VectorDatabase = VectorDatabase(provider)
        db_conn = await vector_db.connect()
        return LongTermMemoryIndex(db_conn)

    async def memorize(self, message: MemorizedMessage):
        await self._db_conn.index(
            VectorDatabaseConnection.Indexes.KNOWLEDGE,
            VectorDatabaseConnection.DBEntry(
                message.message_id,
                 {"type": "memory"},
                message.text, 
            )
        )

    async def get_closest_messages(self, reference: str, *, n=5) -> list:
        return await self._db_conn.search(VectorDatabaseConnection.Indexes.MEMORIES, reference, n)

class KnowledgeIndex:
    def __init__(self, _db_conn: VectorDatabaseConnection): 
        self._db_conn = _db_conn

    @staticmethod
    async def from_provider(provider: ProviderData) -> "KnowledgeIndex":
        vector_db: VectorDatabase = VectorDatabase(provider)
        db_conn = await vector_db.connect()
        return KnowledgeIndex(db_conn)
    
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

    async def chunk_and_index(self, text: str, *, metadata={"type": "knowledge"}) -> int:
        chunks = KnowledgeIndex.chunk_text(text)
        for chunk in chunks:
            await self._db_conn.index(
                VectorDatabaseConnection.Indexes.KNOWLEDGE,
                VectorDatabaseConnection.DBEntry(
                    int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 10),
                    metadata,
                    chunk, 
                )
            )
        return len(chunks)

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
                n_chunks = await self.chunk_and_index(text)
                return n_chunks

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

    def retrieve(self, related_text: str, n=5):
        return self._db_conn.search(
            VectorDatabaseConnection.Indexes.KNOWLEDGE, 
            related_text, 
            n
        )