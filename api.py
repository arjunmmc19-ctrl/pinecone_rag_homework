"""FastAPI backend: upload a PDF (page-based ingestion into a new vector-DB
collection) and ask questions against any ingested collection, on either of
two providers — Pinecone (the original apple-10k-2025 homework index and
any later upload) or a local, on-disk Qdrant instance.

Run with: uvicorn api:app --reload --port 8000
"""

import tempfile
from pathlib import Path
from typing import List, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import qdrant_core
import rag_core
from rag_core import DEFAULT_INDEX_NAME

load_dotenv()

app = FastAPI(title="Pinecone RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024

Provider = Literal["pinecone", "qdrant"]


class AskRequest(BaseModel):
    question: str
    index_name: str = DEFAULT_INDEX_NAME
    provider: Provider = "pinecone"


class SourcePage(BaseModel):
    page_number: Optional[int] = None
    source: Optional[str] = None
    text: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    sources: List[SourcePage]
    context: Optional[str] = None


class UploadResponse(BaseModel):
    index_name: str
    filename: str
    page_count: int
    provider: Provider = "pinecone"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload(
    file: UploadFile = File(...),
    provider: Provider = Form("pinecone"),
) -> UploadResponse:
    filename = file.filename or "document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / filename
        size = 0
        with open(tmp_path, "wb") as out:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=400, detail="File too large (25 MB limit).")
                out.write(chunk)

        if size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        try:
            if provider == "qdrant":
                collection_name = qdrant_core.make_collection_name(filename)
                page_count = qdrant_core.ingest_pdf(tmp_path, collection_name, source_name=filename)
            else:
                collection_name = rag_core.make_index_name(filename)
                page_count = rag_core.ingest_pdf(tmp_path, collection_name, source_name=filename)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return UploadResponse(
        index_name=collection_name, filename=filename, page_count=page_count, provider=provider
    )


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    try:
        if request.provider == "qdrant":
            result = qdrant_core.ask(request.question, collection_name=request.index_name)
        else:
            result = rag_core.ask(request.question, index_name=request.index_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc
    return AskResponse(**result)
