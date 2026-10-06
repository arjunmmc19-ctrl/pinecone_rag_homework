"""Week 6 homework: Evaluate RAG answers using LLM-as-Judge.

Reuses the existing Apple 10-K Pinecone RAG setup (../rag_core.py) for
retrieval + GPT-4o-mini generation, unchanged. A separate model (Gemini)
then judges each (question, retrieved context, answer) triple and gives a
verdict of exactly "correct" or "incorrect".

Run from anywhere, e.g.:
    python week6_llm_judge/evaluate_rag.py
"""

import json
import os
import sys
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag_core import DEFAULT_INDEX_NAME, build_chain  # noqa: E402

RESULTS_PATH = Path(__file__).resolve().parent / "results.json"
RAG_MODEL_NAME = "gpt-4o-mini"
JUDGE_MODEL_NAME = "gemini-3-flash-preview"

QUESTIONS = [
    "Can you tell me Apple's revenue by products and services for fiscal year 2025?",
    "Where is Apple headquartered?",
    "How many full-time equivalent employees did Apple have as of September 27, 2025?",
    "What is the fiscal year end date covered by this Form 10-K?",
    "Did Apple pay dividends or repurchase shares during fiscal year 2025?",
]

JUDGE_PROMPT = ChatPromptTemplate.from_template(
    "You are an impartial judge evaluating a Retrieval-Augmented Generation "
    "(RAG) system's answer to a question about Apple Inc.'s fiscal year 2025 "
    "Form 10-K.\n\n"
    "Question:\n{question}\n\n"
    "Retrieved context (what the RAG system had available):\n{context}\n\n"
    "Generated answer (to evaluate):\n{answer}\n\n"
    "Judge the answer strictly against the retrieved context: is it factually "
    "supported by the context, and does it correctly answer the question? "
    "Respond with exactly one word on the first line, either `correct` or "
    "`incorrect`, followed by one short sentence explaining why."
)


def run_rag(question: str) -> dict:
    retriever, chain = build_chain(DEFAULT_INDEX_NAME)
    docs = retriever.invoke(question)
    answer = chain.invoke(question)

    context_text = "\n\n".join(
        f"[Page {doc.metadata.get('page_number')}] {doc.page_content}" for doc in docs
    )
    seen = {}
    for doc in docs:
        seen[doc.metadata.get("page_number")] = doc.metadata.get("source")
    sources = [
        {"page_number": page, "source": source}
        for page, source in sorted(seen.items(), key=lambda item: (item[0] is None, item[0]))
    ]
    return {"answer": answer, "context_text": context_text, "sources": sources}


def run_judge(question: str, context_text: str, answer: str) -> dict:
    llm = ChatGoogleGenerativeAI(model=JUDGE_MODEL_NAME, temperature=0)
    chain = JUDGE_PROMPT | llm | StrOutputParser()
    raw = chain.invoke(
        {"question": question, "context": context_text, "answer": answer}
    ).strip()

    lines = raw.splitlines()
    first_line = lines[0].strip().lower() if lines else ""
    verdict = "correct" if first_line.startswith("correct") else "incorrect"
    reasoning = "\n".join(lines[1:]).strip() if len(lines) > 1 else raw
    return {"verdict": verdict, "reasoning": reasoning}


def main() -> None:
    if not os.environ.get("GOOGLE_API_KEY"):
        print(
            "ERROR: GOOGLE_API_KEY is not set in .env.\n"
            "Add a line `GOOGLE_API_KEY=<your Gemini key>` to pinecone_rag_homework/.env "
            "(the same key used as GOOGLE_GENAI_API_KEY in my_test_chat_app/.env.local) "
            "and re-run this script."
        )
        sys.exit(1)

    records = []
    for i, question in enumerate(QUESTIONS, start=1):
        print(f"\n[{i}/{len(QUESTIONS)}] Running RAG for: {question}")
        rag_result = run_rag(question)
        pages = [s["page_number"] for s in rag_result["sources"]]
        print(f"    Retrieved pages: {pages}")

        print(f"    Judging with {JUDGE_MODEL_NAME}...")
        judge_result = run_judge(question, rag_result["context_text"], rag_result["answer"])
        print(f"    Verdict: {judge_result['verdict']}")

        records.append(
            {
                "question": question,
                "rag_model": RAG_MODEL_NAME,
                "rag_answer": rag_result["answer"],
                "sources": rag_result["sources"],
                "judge_model": JUDGE_MODEL_NAME,
                "judge_reasoning": judge_result["reasoning"],
                "verdict": judge_result["verdict"],
            }
        )

    RESULTS_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("WEEK 6 LLM-AS-JUDGE EVALUATION SUMMARY")
    print("=" * 80)
    for i, record in enumerate(records, start=1):
        pages = [s["page_number"] for s in record["sources"]]
        print(f"\n[{i}] Q: {record['question']}")
        print(f"    A ({record['rag_model']}): {record['rag_answer']}")
        print(f"    Source pages: {pages}")
        print(f"    Judge ({record['judge_model']}): {record['verdict'].upper()}")
        print(f"    Reasoning: {record['judge_reasoning']}")

    correct_count = sum(1 for r in records if r["verdict"] == "correct")
    print(f"\nResult: {correct_count}/{len(records)} answers judged correct.")
    print(f"Saved detailed results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
