import os
import tempfile
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# function
def load_and_chunk(
        file_bytes: bytes,
        filename: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100
) -> List[Document]:
    """
    Accept raw file bytes (from Streamlit uploader),
    save to a temp file, load, chunk and return documents.
    """

    suffix = ".pdf" if filename.lower().endswith (".pdf") else ".txt"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path, encoding="utf-8")

        documents = loader.load()

        # 
        for doc in documents:
            doc.metadata["source"] = filename

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return splitter.split_documents(documents)
    
    finally:
        os.remove(tmp_path)