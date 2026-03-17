import streamlit as st
import sys
import tempfile

# LangChain (OLD stable API - compatible with versions above)
from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.tools import DuckDuckGoSearchResults

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="AI PDF Chatbot", layout="wide")

st.title("📄 AI PDF + Web Chatbot")

# Debug info
st.sidebar.title("⚙️ Debug")
st.sidebar.code(sys.version)

# ----------------------------
# API KEY
# ----------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ Please add GROQ_API_KEY in Streamlit secrets")
    st.stop()

# LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama3-70b-8192"
)

# ----------------------------
# SESSION STATE
# ----------------------------
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# ----------------------------
# FILE UPLOAD (FIXED KEY)
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
            documents.extend(loader.load())

    st.success(f"✅ Loaded {len(documents)} pages")

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    split_docs = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings()

    # Vector DB
    vector_db = Chroma.from_documents(split_docs, embedding=embeddings)
    st.session_state.vector_db = vector_db

    st.success("✅ PDF processed successfully!")

# ----------------------------
# OPTIONS
# ----------------------------
use_web = st.checkbox("🌐 Enable Web Search", key="web_checkbox")

# ----------------------------
# QUERY INPUT (FIXED KEY)
# ----------------------------
query = st.text_input("Ask your question:", key="query_input")

# ----------------------------
# QUERY HANDLING
# ----------------------------
if query:
    final_output = ""

    # PDF QA
    if st.session_state.vector_db is not None:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=st.session_state.vector_db.as_retriever()
        )

        try:
            pdf_answer = qa_chain.run(query)
            final_output += f"### 📄 PDF Answer\n{pdf_answer}\n\n"
        except Exception as e:
            st.error(f"PDF Error: {e}")

    # Web search
    if use_web:
        try:
            search_tool = DuckDuckGoSearchResults()
            web_result = search_tool.run(query)
            final_output += f"### 🌐 Web Results\n{web_result}"
        except Exception as e:
            st.error(f"Web Error: {e}")

    # Output
    if final_output:
        st.markdown(final_output)
    else:
        st.warning("⚠️ Please upload a PDF or enable web search")
