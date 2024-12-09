import re
import os
import glob
import asyncio

from bot_workflow.vector_db import VectorDatabase
from ai_apis.providers import ProviderData
from concurrent.futures import ThreadPoolExecutor, as_completed

# TODO: abstract this further
class LongTermMemoryIndex:
    def __init__(self, provider: ProviderData): 
        self.vector_db: VectorDatabase = VectorDatabase(provider)

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
            await self.vector_db.index(
                data=chunk, 
                metadata=metadata,
                entry_id=None
            )

    async def index_from_folder(self, path, max_workers=8):
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

        async def process_chunk_group(chunks, metadata="knowledge"):
            for chunk in chunks:
                await self.vector_db.index(
                    data=chunk,
                    metadata=metadata,
                    entry_id=None
                )

        async def process_file(file_path):
            with open(file_path, 'r') as file:
                text = file.read()
                chunks = KnowledgeIndex.chunk_text(text)
                chunk_groups = [chunks[i::max_workers] for i in range(max_workers)]

                tasks = [process_chunk_group(group) for group in chunk_groups]
                await asyncio.gather(*tasks)
                return len(chunks)

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(loop.run_until_complete, process_file(file)): file for file in txt_files}

            total_chunks = 0
            completed_chunks = 0

            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    chunks_in_file = future.result()
                    total_chunks += chunks_in_file
                    completed_chunks += chunks_in_file
                    print(f"Indexed {file_path}")
                    print(f"Progress: {completed_chunks}/{total_chunks} chunks indexed.")
                except Exception as e:
                    print(f"Error indexing {file_path}: {e}")

    async def retrieve(self, related_text):
        return await self.vector_db.search(related_text)