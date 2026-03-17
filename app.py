# ----------------------------
# DISABLE TELEMETRY (FIX WARNINGS)
# ----------------------------
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_IMPL"] = "none"

# ----------------------------
# IMPORTS
# ----------------------------
import streamlit as st
import tempfile
import shutil
import time

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.tools import DuckDuckGoSearchRun

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="AI Chatbot", layout="wide")

# ----------------------------
# SAFE CSS (NO SYNTAX ERROR)
# ----------------------------
st.markdown(
    """
    <style>
    .chat-user {
        background-color: #4CAF50;
        color: white;
        padding: 12px;
        border-radius: 12px;
        margin: 8px;
    }
    .chat-bot {
        background-color: #f1f0f0;
        padding: 12px;
        border-radius: 12px;
        margin: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🤖 AI PDF + Web Chatbot")

# ----------------------------
# API KEY
# ----------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ Add GROQ_API_KEY in Streamlit secrets")
    st.stop()

# ----------------------------
# LLM
# ----------------------------
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"
)

# ----------------------------
# SESSION STATE
# ----------------------------
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "sources" not in st.session_state:
    st.session_state.sources = []

# ----------------------------
# FILE UPLOAD
# ----------------------------
uploaded_files = st.file_uploader(
    "📄 Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    key="pdf_uploader_unique"
)

# ----------------------------
# PROCESS PDF
# ----------------------------
if uploaded_files and st.session_state.vector_db is None:

    docs = []
    names = []

    for file in uploaded_files:
        names.append(file.name)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())

            loader = PyPDFLoader(tmp.name)
            docs.extend(loader.load())

    st.success(f"✅ Loaded: {', '.join(names)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    split_docs = splitter.split_documents(docs)

    embeddings = FakeEmbeddings(size=384)

    # FIX CHROMA ERROR
    persist_dir = "chroma_db"
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    vector_db = Chroma.from_documents(
        split_docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    st.session_state.vector_db = vector_db
    st.success("✅ PDF processed!")

# ----------------------------
# OPTIONS
# ----------------------------
use_web = st.sidebar.checkbox("🌐 Enable Web Search")

# ----------------------------
# SHOW CHAT
# ----------------------------
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"<div class='chat-user'>🧑 {msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bot'>🤖 {msg}</div>", unsafe_allow_html=True)

# ----------------------------
# INPUT
# ----------------------------
query = st.chat_input("Ask something...")

# ----------------------------
# HANDLE QUERY
# ----------------------------
if query:

    st.session_state.chat_history.append(("user", query))

    response = ""
    sources = []

    # PDF ANSWER
    if st.session_state.vector_db:
        try:
            retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 3})

            docs = retriever.get_relevant_documents(query)

            for d in docs:
                sources.append(d.page_content[:200])

            qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

            result = qa.run(query)
            response += f"📄 {result}\n\n"

        except Exception as e:
            response += f"PDF Error: {str(e)}\n"

    # WEB SEARCH
    if use_web:
        try:
            search = DuckDuckGoSearchRun()
            web = search.run(query)
            response += f"🌐 {web}"
        except Exception as e:
            response += f"Web Error: {str(e)}"

    if response == "":
        response = "⚠️ No results found."

    # ----------------------------
    # STREAM RESPONSE
    # ----------------------------
    placeholder = st.empty()
    temp = ""

    for ch in response:
        temp += ch
        placeholder.markdown(
            f"<div class='chat-bot'>🤖 {temp}</div>",
            unsafe_allow_html=True
        )
        time.sleep(0.003)

    st.session_state.chat_history.append(("bot", response))
    st.session_state.sources = sources

    st.rerun()

# ----------------------------
# SOURCES
# ----------------------------
if st.session_state.sources:
    st.sidebar.markdown("### 📚 Sources")
    for i, s in enumerate(st.session_state.sources):
        st.sidebar.write(f"{i+1}. {s[:150]}...")

# ----------------------------
# CLEAR CHAT
# ----------------------------
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.sources = []
    st.session_state.vector_db = None
    st.rerun()
