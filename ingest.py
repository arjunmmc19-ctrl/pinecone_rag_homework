"""CLI: load the Apple 10-K PDF (one page = one Document) and upsert it into
Pinecone. Ingestion logic lives in rag_core.ingest_pdf so it's shared with
the FastAPI upload endpoint used for other PDFs.
"""

from pathlib import Path

from dotenv import load_dotenv

from rag_core import DEFAULT_INDEX_NAME, ingest_pdf

load_dotenv()

PDF_PATH = Path("data/_10-K-2025-As-Filed.pdf")


def main() -> None:
    print(f"Ingesting {PDF_PATH.name} into index '{DEFAULT_INDEX_NAME}'...")
    count = ingest_pdf(PDF_PATH, DEFAULT_INDEX_NAME, id_prefix="apple-10k")
    print(f"Loaded {count} pages from {PDF_PATH.name}")
    print(f"Upserted {count} page-vectors into index '{DEFAULT_INDEX_NAME}'")


if __name__ == "__main__":
    main()
