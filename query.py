"""Ask questions against the Apple 10-K page-index stored in Pinecone."""

import os
import sys

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "apple-10k-2025")

PROMPT = ChatPromptTemplate.from_template(
    "You are answering questions about Apple's 2025 Form 10-K filing.\n"
    "Use only the context below to answer. Cite the page number(s) you used.\n"
    "If the answer isn't in the context, say you don't know.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

REQUIRED_QUESTIONS = [
    "Can you tell me Apple revenue by products and services?",
    "Where is Apple headquarters?",
]


def format_docs(docs) -> str:
    return "\n\n".join(
        f"[Page {doc.metadata.get('page_number')}]\n{doc.page_content}" for doc in docs
    )


def build_chain():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return retriever, chain


def ask(question: str, retriever, chain) -> str:
    answer = chain.invoke(question)
    pages = sorted({doc.metadata.get("page_number") for doc in retriever.invoke(question)})
    return f"{answer}\n\n(Retrieved from pages: {pages})"


def main() -> None:
    retriever, chain = build_chain()

    args_question = " ".join(sys.argv[1:]).strip()
    questions = [args_question] if args_question else REQUIRED_QUESTIONS

    for question in questions:
        print(f"\nQ: {question}")
        print(ask(question, retriever, chain))


if __name__ == "__main__":
    main()
