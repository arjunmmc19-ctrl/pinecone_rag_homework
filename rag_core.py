"""Shared Pinecone/LangChain helpers used by ingest.py, query.py, and api.py.

Every ingestion path (the Apple 10-K CLI or an uploaded PDF) goes through the
same `ingest_pdf` function, so page-based chunking (one Document per PDF
page, with page/page_number/source metadata) is identical everywhere.
"""

import os
import re
import time
import uuid
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Centralized here (rather than in each caller) so PINECONE_INDEX_NAME below
# is read correctly regardless of import order in ingest.py/query.py/api.py.
load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
CHAT_MODEL = "gpt-4o-mini"
DEFAULT_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "apple-10k-2025")
RETRIEVER_K = 10
MAX_INDEX_NAME_LENGTH = 45

PROMPT = ChatPromptTemplate.from_template(
    "You are answering questions about the document below.\n"
    "Use only the context below to answer. Cite the page number(s) you used.\n"
    "If the answer isn't in the context, say you don't know.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)


def get_pinecone_client() -> Pinecone:
    return Pinecone(api_key=os.environ["PINECONE_API_KEY"])


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def make_index_name(filename: str) -> str:
    """Derive a Pinecone-safe, unique index name from an uploaded filename.

    Pinecone index names must be lowercase alphanumeric/hyphen and <= 45
    chars. A slug of the filename keeps it recognizable; the random suffix
    keeps repeated uploads of the same filename from colliding.
    """
    base = re.sub(r"[^a-z0-9]+", "-", Path(filename).stem.lower()).strip("-")
    base = base[:20] or "doc"
    suffix = uuid.uuid4().hex[:8]
    return f"{base}-{suffix}"[:MAX_INDEX_NAME_LENGTH]


def ensure_index(pc: Pinecone, index_name: str, dimension: int = EMBEDDING_DIMENSION) -> None:
    existing = {idx["name"] for idx in pc.list_indexes()}
    if index_name in existing:
        return
    cloud = os.environ.get("PINECONE_CLOUD", "aws")
    region = os.environ.get("PINECONE_REGION", "us-east-1")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud=cloud, region=region),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)


def load_pages(pdf_path: Path, source_name: Optional[str] = None) -> List:
    """Load a PDF as one LangChain Document per page, with page metadata."""
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    name = source_name or pdf_path.name
    for doc in pages:
        doc.metadata["source"] = name
        doc.metadata["page_number"] = doc.metadata["page"] + 1
    return pages


def ingest_pdf(
    pdf_path: Path,
    index_name: str,
    source_name: Optional[str] = None,
    id_prefix: Optional[str] = None,
) -> int:
    """Load a PDF page-by-page and upsert one vector per page into Pinecone.

    Returns the number of pages ingested. `id_prefix` lets callers keep a
    stable vector-id scheme (e.g. the original apple-10k CLI ids) so re-runs
    overwrite rather than duplicate; it defaults to `index_name`, which is
    always unique per upload.
    """
    pc = get_pinecone_client()
    ensure_index(pc, index_name)

    pages = load_pages(pdf_path, source_name=source_name)
    embeddings = get_embeddings()
    prefix = id_prefix or index_name
    ids = [f"{prefix}-page-{doc.metadata['page_number']}" for doc in pages]

    PineconeVectorStore.from_documents(
        documents=pages,
        embedding=embeddings,
        index_name=index_name,
        ids=ids,
    )
    return len(pages)


def format_docs(docs) -> str:
    return "\n\n".join(
        f"[Page {doc.metadata.get('page_number')}]\n{doc.page_content}" for doc in docs
    )


def build_chain(index_name: str, k: int = RETRIEVER_K):
    embeddings = get_embeddings()
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return retriever, chain


def ask(question: str, index_name: str = DEFAULT_INDEX_NAME, k: int = RETRIEVER_K) -> dict:
    retriever, chain = build_chain(index_name, k=k)
    answer = chain.invoke(question)
    docs = retriever.invoke(question)
    seen = {}
    for doc in docs:
        page = doc.metadata.get("page_number")
        seen[page] = doc.metadata.get("source")
    sources = [
        {"page_number": page, "source": source}
        for page, source in sorted(seen.items(), key=lambda item: (item[0] is None, item[0]))
    ]
    return {"answer": answer, "sources": sources}
