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

# Page config
st.set_page_config(page_title="AI PDF Chatbot", layout="wide")

st.title("📄 AI PDF + Web Chatbot")

# Debug
st.sidebar.title("⚙️ Debug")
st.sidebar.code(sys.version)

# API Key
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Add GROQ_API_KEY in secrets")
    st.stop()

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama3-70b-8192"
)

# Session state
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# File uploader
uploaded_files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    key="pdf_uploader"
)

# Process PDFs
if uploaded_files:
    documents = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            loader = PyPDFLoader(tmp_file.name)
            docs = loader.load()
            documents.extend(docs)

    st.success(f"Loaded {len(documents)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    split_docs = splitter.split_documents(documents)

    embeddings = FakeEmbeddings(size=384)

    vector_db = Chroma.from_documents(split_docs, embedding=embeddings)
    st.session_state.vector_db = vector_db

    st.success("PDF processed successfully")

# Web toggle
use_web = st.checkbox("Enable Web Search", key="web_checkbox")

# Query input
query = st.text_input("Ask your question:", key="query_input")

# Handle query
if query:
    output = ""

    if st.session_state.vector_db is not None:
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=st.session_state.vector_db.as_retriever()
            )

            pdf_answer = qa_chain.run(query)
            output += "### PDF Answer\n" + pdf_answer + "\n\n"

        except Exception as e:
            st.error(str(e))

    if use_web:
        try:
            search = DuckDuckGoSearchResults()
            web_result = search.run(query)
            output += "### Web Results\n" + web_result

        except Exception as e:
            st.error(str(e))

    if output:
        st.markdown(output)
    else:
        st.warning("Upload PDF or enable web search")
