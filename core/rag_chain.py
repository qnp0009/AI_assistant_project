"""
RAG Chain Module - Hybrid mode using AI-based routing
"""

import requests
from config.config import LLM_API_URL, OPENAI_API_KEY
from core.milvus_utilis import collection, search_similar_chunks


# Gửi prompt tùy chỉnh đến LLM
def ask_llm_with_context_custom_prompt(prompt: str) -> str:
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. You must only use the provided document content to answer."},
            {"role": "user", "content": prompt}
        ]
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(LLM_API_URL, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# Semantic search
def ask_llm_with_context(query: str) -> str:
    results = search_similar_chunks(query, top_k=1000)
    context = "\n".join([r["chunk"] for r in results])

    prompt = f"""
You are a helpful assistant with access to document snippets.

Based on the following context, answer the user's question concisely and clearly:

CONTEXT:
{context}

QUESTION:
{query}
"""
    return ask_llm_with_context_custom_prompt(prompt)


# Full-context approach: đọc toàn bộ dữ liệu
def load_all_chunks_for_context(limit: int = 9999) -> str:
    collection.load()
    results = collection.query(
        expr="",
        output_fields=["chunk"],
        limit=limit
    )
    chunks = [r["chunk"] for r in results]
    return "\n".join(chunks)

def ask_with_full_context(query: str) -> str:
    full_data = load_all_chunks_for_context(limit=9999)

    prompt = f"""
You are a curious, creative AI assistant.

Below is a collection of document excerpts from various articles and reports:

{full_data}

Now, based on this user question: "{query}", please extract and explain something surprisingly interesting, insightful, or fun. Be creative. Do not repeat the same structure every time. Express it in a friendly and engaging tone.
"""
    return ask_llm_with_context_custom_prompt(prompt)


# Dùng LLM để chọn chiến lược: "search" hoặc "fullcontext"
def route_query_strategy(query: str) -> str:
    prompt = f"""
As an AI assistant, your job is to decide the best strategy to answer the user's question.

You have two tools:

1. search → for specific, factual questions (e.g. "What is the privacy clause?")
2. fullcontext → for vague, open-ended, or creative questions (e.g. "Tell me something cool")

Respond only with: "search" or "fullcontext"

Question: "{query}"
"""
    result = ask_llm_with_context_custom_prompt(prompt)
    return result.strip().lower()


# Hàm duy nhất gọi từ GUI
def ask_question_smart(query: str) -> str:
    strategy = route_query_strategy(query)
    print(strategy)
    if "full" in strategy:
        return ask_with_full_context(query)
    else:
        return ask_llm_with_context(query)
