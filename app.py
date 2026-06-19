import os
import streamlit as st
from dotenv import load_dotenv

from src.loader import load_and_chunk
from src.vectorstore import build_vectorestore, load_vectorstore, vectorestore_exists
from src.rag import build_rag_chain

load_dotenv()

st.set_page_config(
    page_title="RAG Document Intelligence",
    page_icon="📚",
    layout="wide",
)

# Initialise session_state keys
# These persist across reruns. Set default values only if key doesn't exist yet.
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []      # chat history: list of {role, content} dicts
if "doc_loaded" not in st.session_state:
    st.session_state.doc_loaded = False

# sidebars
with st.sidebar:
    st.title("⚙️ Setup")

    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv(" ", ""), # groq api key
        help="Get a free key at console.groq.com"
    )

    st.divider()
    st.subheader("📄 Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF or TXT file",
        type=["pdf", "txt"],
    )

     # Option A: process a newly uploaded file
    if uploaded_file and st.button("🔄 Index Document", type="primary"):
        if not groq_api_key:
            st.error("Please enter your Groq API key first.")
        else:
            with st.spinner("Chunking document..."):
                chunks = load_and_chunk(
                    file_bytes=uploaded_file.read(),
                    filename=uploaded_file.name,
                )
            st.success(f"✅ {len(chunks)} chunks created")

            with st.spinner("Embedding and saving to ChromaDB..."):
                st.session_state.vectorstore = build_vectorestore(chunks)

            with st.spinner("Building RAG chain..."):
                st.session_state.chain, _ = build_rag_chain(
                    st.session_state.vectorstore,
                    groq_api_key
                )

            st.session_state.doc_loaded = True
            st.session_state.messages = []    # clear chat on new document
            st.success(f"✅ Ready — ask questions about {uploaded_file.name}")

    # Option B: resume from an existing ChromaDB on disk
    if vectorestore_exists() and not st.session_state.doc_loaded:
        if st.button("📂 Load existing index"):
            if not groq_api_key:
                st.error("Please enter your Groq API key first.")
            else:
                with st.spinner("Loading ChromaDB from disk..."):
                    st.session_state.vectorstore = load_vectorstore()
                    st.session_state.chain, _ = build_rag_chain(
                        st.session_state.vectorstore,
                        groq_api_key
                    )
                st.session_state.doc_loaded = True
                st.success("✅ Existing index loaded")

    # Show status
    st.divider()
    if st.session_state.doc_loaded:
        count = st.session_state.vectorstore._collection.count()
        st.success(f"🟢 Index active — {count} chunks")
    else:
        st.warning("🔴 No index loaded")

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# main content

st.title("📚 RAG Document Intelligence")
st.caption("Upload a document, then ask questions about it.")

# Replay all previous messages from session_state
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input — only active when a document is indexed
if st.session_state.doc_loaded:
    if user_input := st.chat_input("Ask a question about your document..."):

        # Show and store the user message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Generate and show the assistant answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.chain.invoke(user_input)
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.chat_input("Upload and index a document first...", disabled=True)
