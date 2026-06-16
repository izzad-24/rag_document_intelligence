import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configuration variables
FILE        = "data/pdf/sample.pdf"
CHROMA_DIR  = "./chroma_db"          # where ChromaDB saves to disk
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# loading model
print("Loading embedding model...\n")
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# loading and chunking document
print("Loading and chunking document...\n")
if os.path.exists(CHROMA_DIR):
    print("Existing ChromaDB found — loading from disk...")
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
else:
    print("No ChromaDB found — building from scratch...")

    # load document
    loader = PyPDFLoader(FILE)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from {FILE}\n")

    # chunk document
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from the loaded documents\n")

    # create vector store
    print("Creating ChromaDB vector store from chunks...\n")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
        
    )
    print(f"Persisting ChromaDB to {CHROMA_DIR}\n")

# inspect the vector store
collection = vectorstore._collection
print(f"\nTotal vectors stored: {collection.count()}")

# run similarity search test
query = "What is the minimum CGPA required for entry?"
print(f"\nQuery: '{query}'")
print("-" * 50)

results = vectorstore.similarity_search_with_score(query, k=4)
for i, (doc, score) in enumerate(results):
    print(f"\n[Result {i+1}]  similarity score: {score:.4f}")
    print(f"  Page    : {doc.metadata.get('page_label', 'N/A')}")
    print(f"  Source  : {doc.metadata.get('source')}")
    print(f"  Content : {doc.page_content[:250]}")