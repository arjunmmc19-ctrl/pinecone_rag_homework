"""CLI: ask questions against the Apple 10-K page-index stored in Pinecone.

Retrieval/answer logic lives in rag_core.ask so it's shared with the FastAPI
/ask endpoint used for other uploaded PDFs.
"""

import sys

from dotenv import load_dotenv

from rag_core import DEFAULT_INDEX_NAME, ask

load_dotenv()

REQUIRED_QUESTIONS = [
    "Can you tell me Apple revenue by products and services?",
    "Where is Apple headquarters?",
]


def main() -> None:
    args_question = " ".join(sys.argv[1:]).strip()
    questions = [args_question] if args_question else REQUIRED_QUESTIONS

    for question in questions:
        print(f"\nQ: {question}")
        result = ask(question, index_name=DEFAULT_INDEX_NAME)
        pages = [s["page_number"] for s in result["sources"]]
        print(f"{result['answer']}\n\n(Retrieved from pages: {pages})")


if __name__ == "__main__":
    main()
