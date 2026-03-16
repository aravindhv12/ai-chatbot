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
# API KEY
# -----------------------------

groq_api_key = st.sidebar.text_input(
    "Enter Groq API Key",
    type="password",
    key="groq_key"
)

# -----------------------------
# FILE UPLOAD
# -----------------------------

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True,
    key="pdf_upload"
)

# -----------------------------
# USER QUERY
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
# BUILD VECTOR DB
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
# WEB SEARCH
# -----------------------------

search_tool = DuckDuckGoSearchResults()


# -----------------------------
# RUN QUERY
# -----------------------------

if query and groq_api_key:

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama3-8b-8192"
    )

    context = ""

    # PDF SEARCH
    if uploaded_files:

        vector_db = build_vector_db(uploaded_files)

        retriever = vector_db.as_retriever()

        docs = retriever.invoke(query)

        context = "\n\n".join([d.page_content for d in docs])

    # WEB SEARCH
    web_results = search_tool.run(query)

    context = context + "\n\nWeb Results:\n" + web_results


    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question using the context below.

        Context:
        {context}

        Question:
        {question}

        Provide a helpful answer.
        """
    )

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "context": context,
        "question": query
    })

    st.subheader("Answer")

    st.write(response)


elif query and not groq_api_key:

    st.warning("Please enter your Groq API key"))

vector_db = None

if uploaded_files:

    all_docs = []

    for uploaded_file in uploaded_files:

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

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

    st.success("PDFs processed successfully")


# WEB SEARCH TOOL
search = DuckDuckGoSearchResults()


# PROMPT
prompt = ChatPromptTemplate.from_template(
"""
Answer the question using the context below.

Context:
{context}

Question:
{question}
"""
)


query = st.text_input("Ask a question")


if query:

    # WEB SEARCH MODE
    if "search" in query.lower():

        result = search.invoke(query)

        st.write("🌐 Web Search Result:")
        st.write(result)

    # PDF MODE
    elif vector_db:

        retriever = vector_db.as_retriever()

        docs = retriever.invoke(query)

        context = "\n\n".join([d.page_content for d in docs])

        chain = prompt | llm | StrOutputParser()

        with st.spinner("Thinking..."):

            response = chain.invoke({
                "context": context,
                "question": query
            })

        st.write("📄 Answer:")
        st.write(response)

    else:

        st.warning("Upload PDFs or use web search")
documents = splitter.split_documents(docs)

# Vector DB
vector_db = Chroma.from_documents(
    documents,
    embeddings,
    persist_directory="vector_store"
)

retriever = vector_db.as_retriever()

# Prompt
prompt = ChatPromptTemplate.from_template(
"""
Answer the question using the context below.

Context:
{context}

Question:
{question}
"""
)

# Web search tool
search = DuckDuckGoSearchResults()

# Streamlit UI
st.title("AI Research Assistant")

query = st.text_input("Ask a question")

if query:

    if "search" in query.lower():

        result = search.invoke(query)

        st.write("🌐 Web Search Result:")
        st.write(result)

    else:

        docs = retriever.invoke(query)

        context = "\n\n".join([d.page_content for d in docs])

        chain = prompt | llm | StrOutputParser()

        with st.spinner("Thinking..."):
            response = chain.invoke({
                "context": context,
                "question": query
            })

        st.write("📄 Answer:")
        st.write(response)
