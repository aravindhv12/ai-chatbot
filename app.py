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


st.title("AI Research Assistant")


# GROQ LLM
llm = ChatGroq(
    model="llama3-70b-8192",
    api_key=st.secrets["GROQ_API_KEY"]
)

# EMBEDDINGS
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# FILE UPLOAD
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

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
