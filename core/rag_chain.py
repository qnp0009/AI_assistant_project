import requests
from core.milvus_utilis import search_similar_chunks
from config.config import LLM_API_URL, OPENAI_API_KEY

def retrieve_relevant_chunks(query: str, top_k: int = 1000) -> str:
    results = search_similar_chunks(query, top_k)
    return "\n".join([item["chunk"] for item in results])

def ask_llm_with_context(query: str) -> str:
    context = retrieve_relevant_chunks(query)
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for reading legal documents."},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"}
        ]
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(LLM_API_URL, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
