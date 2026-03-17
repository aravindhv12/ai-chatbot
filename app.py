# ----------------------------
# FIX TELEMETRY
# ----------------------------
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_IMPL"] = "none"

# ----------------------------
# IMPORTS
# ----------------------------
import streamlit as st
import sys
import tempfile
import shutil
import time

from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import FakeEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.tools import DuckDuckGoSearchRun

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="AI Chatbot", layout="wide")

# ----------------------------
# UI STYLING (SAFE)
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
# DEBUG
# ----------------------------
st.sidebar.title("⚙️ Debug")
st.sidebar.code(sys.version)

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
# FILE UPLOADER
# ----------------------------
uploaded_files = st.file_uploader(
    "📄 Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    key="pdf_upload_unique"
)

# ----------------------------
# PROCESS PDF
# ----------------------------
if uploaded_files and st.session_state.vector_db is None:

    documents = []
    file_names = []

    for file in uploaded_files:
        file_names.append(file.name)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())

            loader = PyPDFLoader(tmp.name)
            documents.extend(loader.load())

    st.success(f"✅ Loaded PDFs: {', '.join(file_names)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    split_docs = splitter.split_documents(documents)

    embeddings = FakeEmbeddings(size=384)

    # Safe reset
    persist_dir = "chroma_db"
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    vector_db = Chroma.from_documents(
        split_docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    st.session_state.vector_db = vector_db
    st.success("✅ PDF processed successfully!")

# ----------------------------
# OPTIONS
# ----------------------------
use_web = st.sidebar.checkbox("🌐 Enable Web Search", value=False)

# ----------------------------
# DISPLAY CHAT
# ----------------------------
for role, message in st.session_state.chat_history:
    if role == "user":
        st.markdown(
            f"<div class='chat-user'>🧑 {message}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='chat-bot'>🤖 {message}</div>",
            unsafe_allow_html=True
        )

# ----------------------------
# USER INPUT
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

            for doc in docs:
                sources.append(doc.page_content[:200])

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever
            )

            pdf_answer = qa_chain.run(query)
            response += f"📄 {pdf_answer}\n\n"

        except Exception as e:
            response += f"PDF Error: {str(e)}\n"

    # WEB SEARCH
    if use_web:
        try:
            search = DuckDuckGoSearchRun()
            web_result = search.run(query)
            response += f"🌐 {web_result}"
        except Exception as e:
            response += f"Web Error: {str(e)}"

    if not response:
        response = "⚠️ No results found."

    # ----------------------------
    # STREAMING RESPONSE
    # ----------------------------
    placeholder = st.empty()
    streamed_text = ""

    for char in response:
        streamed_text += char
        placeholder.markdown(
            f"<div class='chat-bot'>🤖 {streamed_text}</div>",
            unsafe_allow_html=True
        )
        time.sleep(0.005)

    st.session_state.chat_history.append(("bot", response))
    st.session_state.sources = sources

    st.rerun()

# ----------------------------
# SHOW SOURCES
# ----------------------------
if st.session_state.sources:
    st.sidebar.markdown("### 📚 PDF Sources")
    for i, src in enumerate(st.session_state.sources):
        st.sidebar.write(f"{i+1}. {src[:150]}...")

# ----------------------------
# CLEAR CHAT
# ----------------------------
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.sources = []
    st.rerun()    color: white;
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
""", unsafe_allow_html=True)

st.title("🤖 AI PDF + Web Chatbot")

# ----------------------------
# DEBUG
# ----------------------------
st.sidebar.title("⚙️ Debug")
st.sidebar.code(sys.version)

# ----------------------------
# API KEY
# ----------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Add GROQ_API_KEY in secrets")
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
# FILE UPLOADER
# ----------------------------
uploaded_files = st.file_uploader(
    "📄 Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    key="pdf_upload"
)

# ----------------------------
# PROCESS PDF
# ----------------------------
if uploaded_files and st.session_state.vector_db is None:

    docs = []
    file_names = []

    for file in uploaded_files:
        file_names.append(file.name)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            loader = PyPDFLoader(tmp.name)
            docs.extend(loader.load())

    st.success(f"✅ Loaded PDFs: {', '.join(file_names)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    split_docs = splitter.split_documents(docs)

    embeddings = FakeEmbeddings(size=384)

    persist_dir = "chroma_db"
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    db = Chroma.from_documents(
        split_docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    st.session_state.vector_db = db
    st.success("✅ PDF Ready!")

# ----------------------------
# OPTIONS
# ----------------------------
use_web = st.sidebar.checkbox("🌐 Enable Web", value=False)

# ----------------------------
# DISPLAY CHAT
# ----------------------------
for role, message in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"<div class='chat-user'>🧑 {message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bot'>🤖 {message}</div>", unsafe_allow_html=True)

# ----------------------------
# USER INPUT
# ----------------------------
query = st.chat_input("Ask something...")

# ----------------------------
# HANDLE QUERY
# ----------------------------
if query:

    st.session_state.chat_history.append(("user", query))

    response = ""
    sources = []

    # PDF ANSWER + SOURCES
    if st.session_state.vector_db:
        try:
            retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(query)

            for doc in docs:
                sources.append(doc.page_content[:200])

            qa = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever
            )

            pdf_ans = qa.run(query)
            response += f"📄 {pdf_ans}\n\n"

        except Exception as e:
            response += f"PDF Error: {e}\n"

    # WEB
    if use_web:
        try:
            search = DuckDuckGoSearchRun()
            web_ans = search.run(query)
            response += f"🌐 {web_ans}"
        except Exception as e:
            response += f"Web Error: {e}"

    if not response:
        response = "No results found."

    # ----------------------------
    # STREAMING EFFECT
    # ----------------------------
    placeholder = st.empty()
    streamed = ""

    for char in response:
        streamed += char
        placeholder.markdown(f"<div class='chat-bot'>🤖 {streamed}</div>", unsafe_allow_html=True)
        time.sleep(0.01)

    st.session_state.chat_history.append(("bot", response))
    st.session_state.sources = sources

    st.rerun()

# ----------------------------
# SHOW SOURCES
# ----------------------------
if st.session_state.sources:
    st.sidebar.markdown("### 📚 Sources (PDF)")
    for i, src in enumerate(st.session_state.sources):
        st.sidebar.write(f"{i+1}. {src[:150]}...")

# ----------------------------
# CLEAR CHAT
# ----------------------------
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.sources = []
    st.rerun()    border-radius: 10px;
    margin: 5px;
}
.chat-bot {
    background-color: #F1F0F0;
    padding: 10px;
    border-radius: 10px;
    margin: 5px;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI PDF + Web Chatbot")

# ----------------------------
# DEBUG
# ----------------------------
st.sidebar.title("⚙️ Debug")
st.sidebar.code(sys.version)

# ----------------------------
# API KEY
# ----------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Add GROQ_API_KEY in secrets")
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

# ----------------------------
# FILE UPLOADER
# ----------------------------
uploaded_files = st.file_uploader(
    "📄 Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    key="pdf_upload"
)

# ----------------------------
# PROCESS PDF
# ----------------------------
if uploaded_files and st.session_state.vector_db is None:

    docs = []

    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            loader = PyPDFLoader(tmp.name)
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    split_docs = splitter.split_documents(docs)

    embeddings = FakeEmbeddings(size=384)

    persist_dir = "chroma_db"
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    db = Chroma.from_documents(
        split_docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    st.session_state.vector_db = db
    st.success("✅ PDF Ready!")

# ----------------------------
# OPTIONS
# ----------------------------
use_web = st.sidebar.checkbox("🌐 Enable Web", value=False)

# ----------------------------
# CHAT DISPLAY
# ----------------------------
for chat in st.session_state.chat_history:
    role, message = chat
    if role == "user":
        st.markdown(f"<div class='chat-user'>🧑 {message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bot'>🤖 {message}</div>", unsafe_allow_html=True)

# ----------------------------
# USER INPUT
# ----------------------------
query = st.chat_input("Ask something...")

# ----------------------------
# HANDLE QUERY
# ----------------------------
if query:

    st.session_state.chat_history.append(("user", query))

    response = ""

    # PDF
    if st.session_state.vector_db:
        try:
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=st.session_state.vector_db.as_retriever()
            )
            pdf_ans = qa.run(query)
            response += f"📄 {pdf_ans}\n\n"
        except Exception as e:
            response += f"PDF Error: {e}\n"

    # WEB
    if use_web:
        try:
            search = DuckDuckGoSearchRun()
            web_ans = search.run(query)
            response += f"🌐 {web_ans}"
        except Exception as e:
            response += f"Web Error: {e}"

    if not response:
        response = "No results found."

    st.session_state.chat_history.append(("bot", response))

    st.rerun()

# ----------------------------
# CLEAR CHAT
# ----------------------------
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()
