from fastapi import FastAPI
from api_embedding import router as embedding_router

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "AI assistant is ready"}

#Đăng ký router

app.include_router(embedding_router,prefix="/embed", tags=["embedding"])