import os
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = "./chroma_db"

def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

def build_vectorestore(chunks: List[Document]) -> Chroma:
    embeddings = get_embeddings()
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

def load_vectorstore() -> Chroma:
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

def vectorestore_exists() -> bool:
    return os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR)