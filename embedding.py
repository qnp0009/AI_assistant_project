from sentence_transformers import SentenceTransformer
from typing import List

# Load mô hình nhúng
model = SentenceTransformer("all-MiniLM-L6-v2")

# Cấu hình chunk
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

def split_into_chunks(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += size - overlap
    return chunks

def embed_chunks(chunks: List[str]) -> List[List[float]]:
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings.tolist()
