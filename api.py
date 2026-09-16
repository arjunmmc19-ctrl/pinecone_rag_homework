"""FastAPI backend: upload a PDF (page-based ingestion into a new Pinecone
index) and ask questions against any ingested index (including the existing
apple-10k-2025 homework index).

Run with: uvicorn api:app --reload --port 8000
"""

import tempfile
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_core import DEFAULT_INDEX_NAME, ask, ingest_pdf, make_index_name

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


class AskRequest(BaseModel):
    question: str
    index_name: str = DEFAULT_INDEX_NAME


class SourcePage(BaseModel):
    page_number: Optional[int] = None
    source: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    sources: List[SourcePage]


class UploadResponse(BaseModel):
    index_name: str
    filename: str
    page_count: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
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

        index_name = make_index_name(filename)
        try:
            page_count = ingest_pdf(tmp_path, index_name, source_name=filename)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return UploadResponse(index_name=index_name, filename=filename, page_count=page_count)


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    try:
        result = ask(request.question, index_name=request.index_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc
    return AskResponse(**result)
