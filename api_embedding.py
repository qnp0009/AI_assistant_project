from fastapi import APIRouter, UploadFile, File, Form, Query
from milvus_utilis import save_to_milvus, search_similar_chunks, collection, delete_file
from embedding import split_into_chunks, embed_chunks
from typing import List
from rag_chain import ask_llm_with_context

router = APIRouter()

@router.post("/text")
async def embed_from_text(text: str = Form(...)):
    chunks = split_into_chunks(text)
    vectors = embed_chunks(chunks)
    return {
        "chunks": chunks,
        "vectors": vectors
    }

@router.post("/file")
async def embed_from_file(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")
    chunks = split_into_chunks(text)
    vectors = embed_chunks(chunks)
    return {
        "filename": file.filename,
        "chunks": chunks,
        "vectors": vectors
    }

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
def delete_file_endpoint(filename: str):
    return delete_file(filename)


    
