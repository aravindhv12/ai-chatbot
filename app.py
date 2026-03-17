# ----------------------------
# DISABLE TELEMETRY (IMPORTANT)
# ----------------------------
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import streamlit as st
import sys
import tempfile

from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import FakeEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.tools import DuckDuckGoSearchResults

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="AI PDF Chatbot", layout="wide")

st.title("📄 AI PDF + Web Chatbot")

# Debug sidebar
st.sidebar.title("⚙️ Debug Info")
st.sidebar.code(sys.version)

# ----------------------------
# API KEY
# ----------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ Please add GROQ_API_KEY in Streamlit secrets")
    st.stop()

# ----------------------------
# LLM (UPDATED MODEL)
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

# ----------------------------
# FILE UPLOADER
# ----------------------------
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    key="pdf_uploader"
)

# ----------------------------
# PROCESS PDF
# ----------------------------
if uploaded_files:
    documents = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            loader = PyPDFLoader(tmp_file.name)
            docs = loader.load()
            documents.extend(docs)

    st.success(f"✅ Loaded {len(documents)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    split_docs = splitter.split_documents(documents)

    # Stable embeddings (no heavy dependencies)
    embeddings = FakeEmbeddings(size=384)

    vector_db = Chroma.from_documents(split_docs, embedding=embeddings)
    st.session_state.vector_db = vector_db

    st.success("✅ PDF processed successfully!")

# ----------------------------
# OPTIONS
# ----------------------------
use_web = st.checkbox("🌐 Enable Web Search", key="web_checkbox")

# ----------------------------
# QUERY INPUT
# ----------------------------
query = st.text_input("Ask your question:", key="query_input")

# ----------------------------
# HANDLE QUERY
# ----------------------------
if query:
    output = ""

    # PDF Answer
    if st.session_state.vector_db is not None:
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=st.session_state.vector_db.as_retriever()
            )

            pdf_answer = qa_chain.run(query)

            if pdf_answer:
                output += "### 📄 PDF Answer\n" + str(pdf_answer) + "\n\n"

        except Exception as e:
            st.error("PDF Error: " + str(e))

    # Web Search
    if use_web:
        try:
            search = DuckDuckGoSearchResults()
            web_result = search.run(query)

            if web_result:
                output += "### 🌐 Web Results\n" + str(web_result)

        except Exception as e:
            st.error("Web Error: " + str(e))

    # FINAL OUTPUT
    if output.strip():
        st.markdown(output)
    else:
        st.warning("⚠️ No results found. Upload PDF or enable web search.")
