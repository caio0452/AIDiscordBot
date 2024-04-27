
import os
import json
import hashlib
from vector_db import QdrantVectorDbConnection

class Knowledge:
    def __init__(self, vector_db_conn: QdrantVectorDbConnection, processed_files_path: str = "processed_knowledge.json"):
        self.processed_files_path = processed_files_path
        self.vector_db_conn = vector_db_conn

    async def start_indexing(self):
        qa_file_path = 'knowledge/qa_knowledge.json'
        if os.path.exists(qa_file_path):
            with open(qa_file_path, 'r') as json_file:
                data = json.load(json_file)
                await self.vector_db_conn.add_qa_knowledge(data)

        KNOWLEDGE_ROOT_DIR = "knowledge"
        paragraphs_to_index: list[str] = []

        if not os.path.exists(KNOWLEDGE_ROOT_DIR):
            os.makedirs(KNOWLEDGE_ROOT_DIR)

        for filename in os.listdir(KNOWLEDGE_ROOT_DIR):
            file_path = os.path.join(KNOWLEDGE_ROOT_DIR, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()

                    if "paragraphs" in filename:
                        paragraphs_to_index.extend(self.split_into_paragraphs(content))
                    elif "chunks" in filename:
                        paragraphs_to_index.extend(self.split_into_chunks(content))

        chunk_size = 64
        for i in range(0, len(paragraphs_to_index), chunk_size):
            chunk = paragraphs_to_index[i:i+chunk_size]
            await self.vector_db_conn.add_text_knowledge(chunk)

        print("Done indexing")

    def split_into_chunks(self, content: str, chunk_size=1000):
        chunks = []
        current_chunk = ""

        words = content.split()
        for word in words:
            if len(current_chunk) + len(word) + 1 > chunk_size:
                chunks.append(current_chunk)
                current_chunk = word
            else:
                if current_chunk:
                    current_chunk += " "
                current_chunk += word

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def split_into_paragraphs(self, content: str, max_paragraph_length=8192*3, min_paragraph_length=10):
        paragraphs_to_index = []
        paragraphs = content.split('\n\n')

        for paragraph in paragraphs:
            if len(paragraph) < min_paragraph_length:
                print(f"Warning: Paragraph '{paragraph}' is too short, skipping.")
            else:
                paragraphs_to_index.append(paragraph)

        return paragraphs_to_index

    def calculate_file_hash(self, file_path: str) -> str:
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as file:
            while True:
                data = file.read(65536)
                if not data:
                    break
                hasher.update(data)
        return hasher.hexdigest()

