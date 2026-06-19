import os
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about a document.
Use ONLY the context provided below. Do not use outside knowledge.
Always cite the page number when referencing specific information.
If the context does not contain the answer, say clearly that you could not find it.

Context:
{context}"""

def _format_docs(docs: List[Document]) -> str:
    formatted = []
    for doc in docs:
        page = doc.metadata.get("page_label", "unknown")
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[Source: {source} | Page: {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

def build_rag_chain(vectorestore: Chroma, groq_api_key: str):
    retriever = vectorestore.as_retriever(
        search_type = 'similarity',
        search_kwargs = {'k': 5}
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])

    llm = ChatGroq(
        api_key=groq_api_key,
        model="llama-3.1-8b-instant", # production model
        temperature=0,   # 0 = deterministic, best for factual document Q&A
    )

    rag_chain = (
        {
            "context": retriever | _format_docs,   # question → chunks → string
            "question": RunnablePassthrough(),    # question passes through unchanged
        }
        | prompt
        | llm
        | StrOutputParser()                       # converts LLM message object → plain string
    )

    return rag_chain, retriever