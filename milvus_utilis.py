from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from embedding import embed_chunks, split_into_chunks
import uuid

# 1. Kết nối tới Milvus
connections.connect(host="localhost", port="19530")

# 2. Định nghĩa schema
fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
    FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
]
schema = CollectionSchema(fields, description="Document Chunks")

# 3. Tạo hoặc lấy Collection
collection = Collection("documents", schema=schema)

if not collection.has_index():  # Kiểm tra đã có index chưa
    collection.create_index(
        field_name="embedding",
        index_params={
            "index_type": "IVF_FLAT",
            "metric_type": "IP",     # Inner Product (càng gần càng tốt)
            "params": {"nlist": 128}
        }
    )

def save_to_milvus(chunks: list[str], filename: str):
    """
    Lưu các đoạn văn và vector tương ứng vào Milvus.
    Mỗi đoạn có 1 id ngẫu nhiên.
    """
    # Generate embeddings for all chunks
    vectors = embed_chunks(chunks)
    
    # Generate unique IDs
    ids = [str(uuid.uuid4()) for _ in chunks]

    # Get filename
    filenames = [filename]*len(chunks)

    # Insert into Milvus
    collection.insert([ids, filenames, chunks, vectors])
    collection.flush()
    print(f"✅ Đã lưu {len(chunks)} đoạn từ file '{filename}' vào Milvus.")

def search_similar_chunks(query: str, top_k: int = 3):
    """
    Nhúng câu hỏi, tìm top_k đoạn văn gần nhất trong Milvus
    """
    # Get query embedding - pass the query as a single-item list
    query_vectors = embed_chunks([query])
    if not query_vectors:
        raise ValueError("Failed to generate embedding for query")
    
    query_vector = query_vectors[0]  # Get first (and only) vector
    collection.load()  # đảm bảo dữ liệu đã sẵn sàng

    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["chunk"]
    )

    matches = []
    for hit in results[0]:
        matches.append({
            "score": hit.score,
            "chunk": hit.entity.get("chunk")
        })

    return matches
