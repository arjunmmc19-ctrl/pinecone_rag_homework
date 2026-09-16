# Apple 10-K RAG (Pinecone + LangChain)

A retrieval-augmented QA app over Apple's 2025 Form 10-K filing. Each PDF page
is loaded as a single LangChain `Document` (page-based chunking, no further
splitting), embedded, and stored in Pinecone with page metadata preserved.

## Files

- `ingest.py` — loads `data/_10-K-2025-As-Filed.pdf` page-by-page, creates the
  Pinecone index if it doesn't exist, and upserts one vector per page.
- `query.py` — builds a retriever + RAG chain over the index and answers
  questions, citing the source page number(s).
- `requirements.txt` — Python dependencies.
- `.env.example` — required environment variable names (no real values).

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in your real API keys:
   ```bash
   cp .env.example .env
   ```
   You'll need:
   - `PINECONE_API_KEY` — from the [Pinecone console](https://app.pinecone.io)
   - `OPENAI_API_KEY` — from the [OpenAI platform](https://platform.openai.com)

## Run

1. Ingest the PDF into Pinecone (creates the index on first run):
   ```bash
   python ingest.py
   ```
2. Ask questions:
   ```bash
   python query.py
   ```
   This runs the two required questions by default:
   - "Can you tell me Apple revenue by products and services?"
   - "Where is Apple headquarters?"

   Or ask your own question:
   ```bash
   python query.py "What was Apple's total net sales in fiscal 2025?"
   ```

## How page-based chunking works

`PyPDFLoader` (from `langchain-community`) returns one `Document` per PDF
page out of the box, with `metadata["page"]` (0-indexed) set automatically.
`ingest.py` adds `metadata["page_number"]` (1-indexed) and `metadata["source"]`
for readability, then upserts each page as its own vector — no
`RecursiveCharacterTextSplitter` or other sub-page splitting is used, per the
assignment's page-based chunking requirement.
