from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from core.embedding import embed_chunks, split_into_chunks
import uuid
import time

print("🕒 Connecting to Milvus...")
start_time = time.time()

# 1. Connect to Milvus
connections.connect(host="localhost", port="19530")

connect_time = time.time() - start_time
print(f"✅ Connected to Milvus in {connect_time:.2f} seconds")

# 2. Define schema
fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
    FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
]
schema = CollectionSchema(fields, description="Document Chunks")

# 3. Create or get Collection
collection = Collection("documents", schema=schema)

if not collection.has_index():  # Check if index exists
    print("🕒 Creating index...")
    index_start = time.time()
    collection.create_index(
        field_name="embedding",
        index_params={
            "index_type": "IVF_SQ8",  # More efficient than IVF_FLAT
            "metric_type": "IP",     # Inner Product (higher is better)
            "params": {"nlist": 1024}  # Increased for better accuracy/speed balance
        }
    )
    index_time = time.time() - index_start
    print(f"✅ Index created in {index_time:.2f} seconds")

def save_to_milvus(chunks: list[str], filename: str, vectors: list[list[float]] | None = None):
    """
    Save text chunks and their vectors to Milvus. Each chunk gets a unique ID.
    """
    start_time = time.time()
    
    # Generate embeddings if not provided
    if vectors is None:
        vectors = embed_chunks(chunks)
    
    # Generate unique IDs
    ids = [str(uuid.uuid4()) for _ in chunks]

    # Prepare filename list
    filenames = [filename]*len(chunks)

    # Insert into Milvus
    collection.insert([ids, filenames, chunks, vectors])
    # Only flush if we have a small number of chunks
    if len(chunks) <= 10:
        collection.flush()
    
    save_time = time.time() - start_time
    print(f"✅ Saved {len(chunks)} chunks to Milvus in {save_time:.2f} seconds")

def search_similar_chunks(query: str, top_k: int = 1000):
    """
    Embed the query and find the top_k most similar text chunks in Milvus.
    """
    start_time = time.time()
    
    # Get query embedding
    query_vectors = embed_chunks([query])
    if not query_vectors:
        raise ValueError("Failed to generate embedding for query")
    
    query_vector = query_vectors[0]  # Use the first (and only) vector
    collection.load()  # Ensure data is loaded

    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 32}},  # Increased nprobe for better accuracy
        limit=top_k,
        output_fields=["chunk"]
    )

    matches = []
    for hit in results[0]:
        matches.append({
            "score": hit.score,
            "chunk": hit.entity.get("chunk")
        })

    search_time = time.time() - start_time
    print(f"⏱️ Search completed in {search_time:.2f} seconds")
    return matches
    
def delete_file(filename: str):
    """
    Delete all chunks of a file from Milvus by filename.
    """
    collection.load()
    collection.delete(expr=f'filename == "{filename}"')
    collection.flush()
    print(f"✅ Deleted all chunks of {filename} from Milvus.")

    return {
        "filename": filename,
        "message": f"✅ Deleted all chunks of {filename} from Milvus."
    }
