"""Load the Apple 10-K PDF (one page = one Document) and upsert it into Pinecone."""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PDF_PATH = Path("data/_10-K-2025-As-Filed.pdf")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "apple-10k-2025")
PINECONE_CLOUD = os.environ.get("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.environ.get("PINECONE_REGION", "us-east-1")


def load_pages() -> list:
    loader = PyPDFLoader(str(PDF_PATH))
    pages = loader.load()
    for doc in pages:
        doc.metadata["source"] = PDF_PATH.name
        doc.metadata["page_number"] = doc.metadata["page"] + 1
    return pages


def ensure_index(pc: Pinecone) -> None:
    existing = {idx["name"] for idx in pc.list_indexes()}
    if INDEX_NAME in existing:
        return
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(1)


def main() -> None:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    ensure_index(pc)

    pages = load_pages()
    print(f"Loaded {len(pages)} pages from {PDF_PATH.name}")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    ids = [f"apple-10k-page-{doc.metadata['page_number']}" for doc in pages]

    PineconeVectorStore.from_documents(
        documents=pages,
        embedding=embeddings,
        index_name=INDEX_NAME,
        ids=ids,
    )
    print(f"Upserted {len(pages)} page-vectors into index '{INDEX_NAME}'")


if __name__ == "__main__":
    main()
