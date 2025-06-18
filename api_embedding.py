from fastapi import APIRouter, UploadFile, File, Form, Query
from milvus_utilis import save_to_milvus, search_similar_chunks, collection
from embedding import split_into_chunks, embed_chunks
from typing import List
from rag_chain import ask_llm_with_context

router = APIRouter()

@router.post("/upload_and_store")
async def upload_and_store(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")

    # Chunk + embedding
    chunks = split_into_chunks(text)
    vectors = embed_chunks(chunks)

    # Lưu vào Milvus with pre-computed vectors
    save_to_milvus(chunks, str(file.filename), vectors)

    return {
        "filename": file.filename,
        "chunks": chunks,
        "vectors": vectors,
        "message": f"✅ Đã lưu {len(chunks)} đoạn vào Milvus."
    }

@router.post("/search")
async def search_chunks(q: str = Query(...)):
    results = search_similar_chunks(q)
    return {
        "query": q,
        "results": results
    }

@router.post("/ask")
def ask(query: str = Form(...)):
    answer = ask_llm_with_context(query)
    return answer

@router.delete("/delete_file_name")
def delete_file(filename: str):
    collection.load()
    collection.delete(expr=f'filename == "{filename}"')
    collection.flush()
    print(f"✅ Đã xóa toàn bộ chunks của {filename} khỏi Milvus.")

    return {
        "filename":filename,
        "message": f"✅ Đã xóa toàn bộ chunks của {filename} khỏi Milvus."
    }


    
