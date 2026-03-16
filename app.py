import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.tools import DuckDuckGoSearchResults


# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(page_title="AI Research Assistant")
st.title("AI Research Assistant")


# -----------------------------
# LOAD GROQ KEY FROM SECRETS
# -----------------------------

groq_api_key = st.secrets["GROQ_API_KEY"]


# -----------------------------
# FILE UPLOAD
# -----------------------------

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True,
    key="pdf_uploader"
)


# -----------------------------
# USER QUESTION
# -----------------------------

query = st.text_input(
    "Ask a question",
    key="query_input"
)


# -----------------------------
# LOAD EMBEDDINGS
# -----------------------------

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# -----------------------------
# BUILD VECTOR DATABASE
# -----------------------------

@st.cache_resource
def build_vector_db(files):

    embeddings = load_embeddings()
    all_docs = []

    for file in files:

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    documents = splitter.split_documents(all_docs)

    vector_db = Chroma.from_documents(
        documents,
        embeddings
    )

    return vector_db


# -----------------------------
# WEB SEARCH TOOL
# -----------------------------

search_tool = DuckDuckGoSearchResults()


# -----------------------------
# RUN QUERY
# -----------------------------

if query:

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama3-8b-8192"
    )

    context = ""

    # -----------------------------
    # PDF SEARCH
    # -----------------------------

    if uploaded_files:

        with st.spinner("Processing PDFs..."):

            vector_db = build_vector_db(uploaded_files)

            retriever = vector_db.as_retriever()

            docs = retriever.invoke(query)

            pdf_context = "\n\n".join([d.page_content for d in docs])

            context += pdf_context


    # -----------------------------
    # WEB SEARCH
    # -----------------------------

    with st.spinner("Searching web..."):

        web_results = search_tool.run(query)

        context += "\n\nWeb Results:\n" + web_results


    # -----------------------------
    # PROMPT
    # -----------------------------

    prompt = ChatPromptTemplate.from_template(
        """
You are an AI research assistant.

Use the context below to answer the question.

Context:
{context}

Question:
{question}

Give a clear helpful answer.
"""
    )


    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "context": context,
        "question": query
    })


    # -----------------------------
    # OUTPUT
    # -----------------------------

    st.subheader("Answer")
    st.write(response)

# -----------------------------
# FILE UPLOADER
# -----------------------------

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True,
    key="pdf_uploader"
)


# -----------------------------
# USER QUESTION
# -----------------------------

query = st.text_input(
    "Ask a question",
    key="query_input"
)


# -----------------------------
# LOAD EMBEDDINGS
# -----------------------------

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# -----------------------------
# BUILD VECTOR DATABASE
# -----------------------------

@st.cache_resource
def build_vector_db(files):

    embeddings = load_embeddings()

    all_docs = []

    for file in files:

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        all_docs.extend(docs)

    # TEXT SPLITTING
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    documents = splitter.split_documents(all_docs)

    # VECTOR DATABASE
    vector_db = Chroma.from_documents(
        documents,
        embeddings
    )

    return vector_db


# -----------------------------
# WEB SEARCH TOOL
# -----------------------------

search_tool = DuckDuckGoSearchResults()


# -----------------------------
# RUN QUERY
# -----------------------------

if query:

    if not groq_api_key:
        st.warning("Please enter Groq API key in sidebar")
        st.stop()

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama3-8b-8192"
    )

    context = ""

    # -----------------------------
    # PDF SEARCH
    # -----------------------------

    if uploaded_files:

        with st.spinner("Processing PDFs..."):

            vector_db = build_vector_db(uploaded_files)

            retriever = vector_db.as_retriever()

            docs = retriever.invoke(query)

            pdf_context = "\n\n".join([d.page_content for d in docs])

            context += pdf_context


    # -----------------------------
    # WEB SEARCH
    # -----------------------------

    with st.spinner("Searching web..."):

        web_results = search_tool.run(query)

        context += "\n\nWeb Results:\n" + web_results


    # -----------------------------
    # PROMPT
    # -----------------------------

    prompt = ChatPromptTemplate.from_template(
        """
You are an AI research assistant.

Use the context below to answer the question.

Context:
{context}

Question:
{question}

Give a clear helpful answer.
"""
    )


    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "context": context,
        "question": query
    })


    # -----------------------------
    # OUTPUT
    # -----------------------------

    st.subheader("Answer")

    st.write(response)
