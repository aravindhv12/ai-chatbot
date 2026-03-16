import os
import streamlit as st

# -------------------------------
# DOCUMENT LOADER
# -------------------------------
from langchain_community.document_loaders import PyPDFLoader

# -------------------------------
# TEXT SPLITTING
# -------------------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------------------
# EMBEDDINGS
# -------------------------------
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------------------
# VECTORSTORE (Chroma)
# -------------------------------
try:
    from langchain.vectorstores import Chroma
except ModuleNotFoundError:
    from langchain.vectorstores.chroma import Chroma

# -------------------------------
# LLM / Groq
# -------------------------------
from langchain_groq import ChatGroq

# -------------------------------
# CONVERSATIONAL RETRIEVAL
# -------------------------------
from langchain.chains import ConversationalRetrievalChain

# -------------------------------
# WEB SEARCH TOOL
# -------------------------------
from langchain_community.tools import DuckDuckGoSearchResults

# -------------------------------
# STREAMLIT SETTINGS
# -------------------------------
st.set_page_config(page_title="Multi-PDF & Web Chatbot", layout="wide")
st.title("📄 Multi-PDF & Web Chatbot with Groq & LangChain")

# -------------------------------
# GROQ API KEY
# -------------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error("Groq API key not found in secrets! Please add it.")
    st.stop()

chat_groq = ChatGroq(api_key=GROQ_API_KEY)

# -------------------------------
# FILE UPLOADER
# -------------------------------
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

docs = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        loader = PyPDFLoader(uploaded_file)
        docs.extend(loader.load())
    st.success(f"Loaded {len(docs)} pages from {len(uploaded_files)} PDF(s).")

# -------------------------------
# WEB SEARCH OPTION
# -------------------------------
use_web_search = st.checkbox("Enable Web Search", value=False)

# -------------------------------
# TEXT SPLITTING
# -------------------------------
split_docs = []
if docs:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

# -------------------------------
# VECTOR DB / EMBEDDINGS
# -------------------------------
vector_db = None
if split_docs:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(split_docs, embedding=embeddings)

# -------------------------------
# QUERY INPUT
# -------------------------------
query = st.text_input("Ask a question")

if query:
    results_text = ""

    # PDF retrieval
    if vector_db:
        retrieval_chain = ConversationalRetrievalChain.from_llm(
            llm=chat_groq,
            retriever=vector_db.as_retriever()
        )
        response = retrieval_chain.run(query)
        results_text += f"**From PDFs:** {response}\n\n"

    # Web search
    if use_web_search:
        search_tool = DuckDuckGoSearchResults()
        search_results = search_tool.run(query)
        results_text += f"**From Web Search:** {search_results}\n\n"

    st.markdown(results_text or "No results found.")# -------------------------------
# WEB SEARCH TOOL
# -------------------------------
from langchain_community.tools import DuckDuckGoSearchResults

# -------------------------------
# STREAMLIT SETTINGS
# -------------------------------
st.set_page_config(page_title="Multi-PDF & Web Chatbot", layout="wide")
st.title("📄 Multi-PDF & Web Chatbot with Groq & LangChain")

# -------------------------------
# GROQ API KEY
# -------------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error("Groq API key not found in secrets! Please add it.")
    st.stop()

chat_groq = ChatGroq(api_key=GROQ_API_KEY)

# -------------------------------
# FILE UPLOADER
# -------------------------------
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

docs = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        loader = PyPDFLoader(uploaded_file)
        docs.extend(loader.load())
    st.success(f"Loaded {len(docs)} pages from {len(uploaded_files)} PDF(s).")

# -------------------------------
# WEB SEARCH OPTION
# -------------------------------
use_web_search = st.checkbox("Enable Web Search", value=False)

# -------------------------------
# TEXT SPLITTING
# -------------------------------
split_docs = []
if docs:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

# -------------------------------
# VECTOR DB / EMBEDDINGS
# -------------------------------
vector_db = None
if split_docs:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(split_docs, embedding=embeddings)

# -------------------------------
# QUERY INPUT
# -------------------------------
query = st.text_input("Ask a question")

if query:
    results_text = ""

    # PDF retrieval
    if vector_db:
        retrieval_chain = ConversationalRetrievalChain.from_llm(
            llm=chat_groq,
            retriever=vector_db.as_retriever()
        )
        response = retrieval_chain.run(query)
        results_text += f"**From PDFs:** {response}\n\n"

    # Web search
    if use_web_search:
        search_tool = DuckDuckGoSearchResults()
        search_results = search_tool.run(query)
        results_text += f"**From Web Search:** {search_results}\n\n"

    st.markdown(results_text or "No results found.")    
    st.stop()

# Initialize Groq
chat_groq = ChatGroq(api_key=GROQ_API_KEY)

# -------------------------------
# UPLOAD PDFs
# -------------------------------
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

docs = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        loader = PyPDFLoader(uploaded_file)
        docs.extend(loader.load())
    st.success(f"Loaded {len(docs)} pages from {len(uploaded_files)} PDF(s).")

# -------------------------------
# WEB SEARCH OPTION
# -------------------------------
use_web_search = st.checkbox("Enable Web Search", value=False)

# -------------------------------
# TEXT SPLITTING
# -------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
split_docs = splitter.split_documents(docs) if docs else []

# -------------------------------
# VECTOR DB / EMBEDDINGS
# -------------------------------
vector_db = None
if split_docs:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_db = Chroma.from_documents(
        split_docs,
        embedding=embeddings
    )

# -------------------------------
# QUERY INPUT
# -------------------------------
query = st.text_input("Ask a question")

if query:
    results_text = ""

    # PDF retrieval
    if vector_db:
        retrieval_chain = ConversationalRetrievalChain.from_llm(
            llm=chat_groq,
            retriever=vector_db.as_retriever()
        )
        response = retrieval_chain.run(query)
        results_text += f"**From PDFs:** {response}\n\n"

    # Web search
    if use_web_search:
        search_tool = DuckDuckGoSearchResults()
        search_results = search_tool.run(query)
        results_text += f"**From Web Search:** {search_results}\n\n"

    st.markdown(results_text or "No results found.")
    docs = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        loader = PyPDFLoader(uploaded_file)
        docs.extend(loader.load())
    st.success(f"Loaded {len(docs)} pages from {len(uploaded_files)} PDF(s).")

# -------------------------------
# WEB SEARCH OPTION
# -------------------------------
use_web_search = st.checkbox("Enable Web Search", value=False)

# -------------------------------
# TEXT SPLITTING
# -------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
split_docs = splitter.split_documents(docs) if docs else []

# -------------------------------
# VECTOR DB / EMBEDDINGS
# -------------------------------
vector_db = None
if split_docs:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_db = Chroma.from_documents(
        split_docs,
        embedding=embeddings
    )

# -------------------------------
# QUERY INPUT
# -------------------------------
query = st.text_input("Ask a question")

if query:
    results_text = ""

    # 1️⃣ PDF Retrieval
    if vector_db:
        retrieval_chain = ConversationalRetrievalChain.from_llm(
            llm=chat_groq,
            retriever=vector_db.as_retriever()
        )
        response = retrieval_chain.run(query)
        results_text += f"**From PDFs:** {response}\n\n"

    # 2️⃣ Web Search
    if use_web_search:
        search_tool = DuckDuckGoSearchResults()
        search_results = search_tool.run(query)
        results_text += f"**From Web Search:** {search_results}\n\n"

    st.markdown(results_text or "No results found.")
