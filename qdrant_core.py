"""Qdrant-backed RAG helpers, mirroring rag_core.py's page-based chunking,
single-retrieval-reuse ask(), and response shape, so api.py can dispatch to
either vector DB behind an identical contract.

Runs against a local, on-disk Qdrant instance (qdrant-client's embedded
mode) rather than a server, so no new API keys or external services are
needed. Qdrant's local mode locks its storage directory to one open client
at a time, so every operation here opens a fresh client and closes it
before returning, rather than holding one open across calls.
"""

import os
import re
import uuid
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
CHAT_MODEL = "gpt-4o-mini"
RETRIEVER_K = 10
MAX_COLLECTION_NAME_LENGTH = 64

QDRANT_PATH = os.environ.get(
    "QDRANT_PATH", str(Path(__file__).resolve().parent / "qdrant_data")
)

PROMPT = ChatPromptTemplate.from_template(
    "You are answering questions about the document below.\n"
    "Use only the context below to answer. Cite the page number(s) you used.\n"
    "If the answer isn't in the context, say you don't know.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)


def get_client() -> QdrantClient:
    return QdrantClient(path=QDRANT_PATH)


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def make_collection_name(filename: str) -> str:
    """Derive a unique, Qdrant-safe collection name from an uploaded
    filename, the same way rag_core.make_index_name does for Pinecone."""
    base = re.sub(r"[^a-z0-9]+", "-", Path(filename).stem.lower()).strip("-")
    base = base[:30] or "doc"
    suffix = uuid.uuid4().hex[:8]
    return f"{base}-{suffix}"[:MAX_COLLECTION_NAME_LENGTH]


def ensure_collection(
    client: QdrantClient, collection_name: str, dimension: int = EMBEDDING_DIMENSION
) -> None:
    if client.collection_exists(collection_name):
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
    )


def load_pages(pdf_path: Path, source_name: Optional[str] = None) -> List:
    """Load a PDF as one LangChain Document per page, with page metadata —
    identical chunking to rag_core.load_pages, so Pinecone and Qdrant
    ingest the same documents the same way."""
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    name = source_name or pdf_path.name
    for doc in pages:
        doc.metadata["source"] = name
        doc.metadata["page_number"] = doc.metadata["page"] + 1
    return pages


def ingest_pdf(
    pdf_path: Path, collection_name: str, source_name: Optional[str] = None
) -> int:
    """Load a PDF page-by-page and upsert one point per page into Qdrant.

    Returns the number of pages ingested.
    """
    pages = load_pages(pdf_path, source_name=source_name)
    embeddings = get_embeddings()

    client = get_client()
    try:
        ensure_collection(client, collection_name)
        vectorstore = QdrantVectorStore(
            client=client, collection_name=collection_name, embedding=embeddings
        )
        vectorstore.add_documents(pages)
    finally:
        client.close()
    return len(pages)


def format_docs(docs) -> str:
    return "\n\n".join(
        f"[Page {doc.metadata.get('page_number')}]\n{doc.page_content}" for doc in docs
    )


def ask(question: str, collection_name: str, k: int = RETRIEVER_K) -> dict:
    """Mirrors rag_core.ask(): retrieve once, generate from that exact
    context, and return that same context alongside the answer — so
    callers (e.g. an LLM-as-judge) see precisely what the generator saw.
    """
    embeddings = get_embeddings()

    client = get_client()
    try:
        vectorstore = QdrantVectorStore(
            client=client, collection_name=collection_name, embedding=embeddings
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(question)
    finally:
        client.close()

    context_text = format_docs(docs)

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    answer_chain = PROMPT | llm | StrOutputParser()
    answer = answer_chain.invoke({"context": context_text, "question": question})

    seen = {}
    for doc in docs:
        page = doc.metadata.get("page_number")
        seen[page] = {"source": doc.metadata.get("source"), "text": doc.page_content}
    sources = [
        {"page_number": page, "source": info["source"], "text": info["text"]}
        for page, info in sorted(seen.items(), key=lambda item: (item[0] is None, item[0]))
    ]
    return {"answer": answer, "context": context_text, "sources": sources}
