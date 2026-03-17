import streamlit as st
import sys
import tempfile

# LangChain imports (UPDATED)
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# New LangChain chain system
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.tools import DuckDuckGoSearchResults

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="AI PDF + Web Chatbot", layout="wide")

# ----------------------------
# DEBUG INFO
# ----------------------------
st.sidebar.title("⚙️ Debug Info")
st.sidebar.write("### Python Version")
st.sidebar.code(sys.version)

# ----------------------------
# TITLE
# ----------------------------
st.title("📄 AI Multi-PDF & Web Chatbot")

# ----------------------------
# GROQ API KEY
# ----------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ Please add GROQ_API_KEY in Streamlit secrets")
    st.stop()

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
# FILE UPLOAD
# ----------------------------
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    docs = []

    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            loader = PyPDFLoader(tmp.name)
            docs.extend(loader.load())

    st.success(f"✅ Loaded {len(docs)} pages")

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    split_docs = splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector DB
    vector_db = Chroma.from_documents(
        split_docs,
        embedding=embeddings
    )

    st.session_state.vector_db = vector_db
    st.success("✅ Vector DB created!")

# ----------------------------
# OPTIONS
# ----------------------------
use_web = st.checkbox("🌐 Enable Web Search")

# ----------------------------
# QUERY INPUT
# ----------------------------
query = st.text_input("Ask a question")

if query:
    result = ""

    # ----------------------------
    # PDF QA (NEW METHOD)
    # ----------------------------
    if st.session_state.vector_db:
        retriever = st.session_state.vector_db.as_retriever()

        prompt = ChatPromptTemplate.from_template(
            "Answer the question based only on the context:\n\n{context}\n\nQuestion: {input}"
        )

        document_chain = create_stuff_documents_chain(llm, prompt)

        qa_chain = create_retrieval_chain(retriever, document_chain)

        response = qa_chain.invoke({"input": query})

        result += f"### 📄 PDF Answer\n{response['answer']}\n\n"

    # ----------------------------
    # WEB SEARCH
    # ----------------------------
    if use_web:
        search = DuckDuckGoSearchResults()
        web_result = search.run(query)

        result += f"### 🌐 Web Results\n{web_result}\n\n"

    # ----------------------------
    # OUTPUT
    # ----------------------------
    if result:
        st.markdown(result)
    else:
        st.warning("⚠️ Upload PDFs or enable web search")
