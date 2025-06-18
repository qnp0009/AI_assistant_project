import os
from dotenv import load_dotenv
load_dotenv()
LLM_API_URL = os.getenv("LLM_API_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not LLM_API_URL:
    raise ValueError("⚠️ Chưa thiết lập LLM_API_URL trong .env")

if not OPENAI_API_KEY:
    raise ValueError("⚠️ Chưa thiết lập OPENAI_API_KEY trong .env")
