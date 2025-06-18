# Vietnamese Document Search with Milvus

This project implements a semantic search system for Vietnamese documents using Milvus vector database and sentence transformers.

## Features

- Document chunking and embedding
- Vector similarity search
- REST API endpoints for document processing and search
- Vietnamese language support

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
Fill out `.env` file with:
```
OPENAI_API_KEY=your_api_key_here
```

3. Download Milvus server:
```bash
Follow the step on google, use the powershell to run the milvus database
```

4. Run the FastAPI server:
```bash
python -m uvicorn main:app --reload
```

## API Endpoints

- `POST /embed/text`: Process text and store embeddings
- `POST /embed/file`: Process file and store embeddings
- `POST /embed/upload_and_store`: Upload the test file and store it into the milvus database
- `POST /embed/search`: Search for similar text chunks
- `POST /embed/ask`: Ask the AI

## Project Structure

- `main.py`: FastAPI application entry point
- `api_embedding.py`: API routes for embedding and search
- `embedding.py`: Text embedding utilities
- `milvus_utilis.py`: Milvus database operations
- `config.py`: Configuration settings 