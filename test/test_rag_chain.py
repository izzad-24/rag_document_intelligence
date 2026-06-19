import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from typing import List

# Configuration variables
load_dotenv()

# Constants
CHROMA_DIR  = "./chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Step 1: Load the ChromaDB vector store
print("Loading ChromaDB...\n")
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectorstore = Chroma(
    persist_directory=CHROMA_DIR, # retrive the data vector database
    embedding_function=embeddings,
)

# Step 2: Create a retriever
print("Creating retriever...\n")
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# Step 3: Define a function to format chunks into readable context
def format_docs(docs: List[Document]) -> str:
    formatted = []
    for doc in docs:
        page = doc.metadata.get("page_label", "?")
        formatted.append(f"[Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


SYSTEM_PROMPT = """You are a helpful assistant that answers questions about a document.
Use ONLY the context provided below to answer. Do not use any outside knowledge.
Always mention the page number when referencing information.
If the context does not contain the answer, say "I could not find the information in the document. Please ask another question."
Context:
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}")
])

# Step 4: Load the LLM 
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant", # production model
    temperature=0,   # 0 = deterministic, best for factual document Q&A
)

# Step 5
rag_chain = (
    {
        "context": retriever | format_docs,   # question → chunks → string
        "question": RunnablePassthrough(),    # question passes through unchanged
    }
    | prompt
    | llm
    | StrOutputParser()                       # converts LLM message object → plain string
)

# test model
questions = [
    "What is the minimum CGPA required for entry?",
    "How many programmes are there?",
    "What are the programme learning outcomes?",
    "What is the weather like today?",
]

for question in questions:
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")
    answer = rag_chain.invoke(question)
    print(answer)