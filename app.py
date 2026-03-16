import streamlit as st
import tempfile
import os

# LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain_community.tools import DuckDuckGoSearchResults

# ----------------------------
# Streamlit Setup
# ----------------------------

st.set_page_config(page_title="AI Research Assistant", page_icon="🤖")

st.title("🤖 AI Research Assistant")
st.write("Search PDFs and the Web")

# ----------------------------
# Upload PDFs
# ----------------------------

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

VECTOR_PATH = "vector_store"

# ----------------------------
# Embeddings
# ----------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# ----------------------------
# Web Search Tool
# ----------------------------

search = DuckDuckGoSearchResults(num_results=3)

# ----------------------------
# Create Vector DB
# ----------------------------

def create_vector_db(files):

    documents = []

    for file in files:

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            filepath = tmp.name

        loader = PyPDFLoader(filepath)
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.split_documents(documents)

    vector_db = FAISS.from_documents(docs, embeddings)

    vector_db.save_local(VECTOR_PATH)

    return vector_db

# ----------------------------
# Load Vector DB
# ----------------------------

def load_vector_db():

    if not os.path.exists("vector_store/index.faiss"):
        return None

    return FAISS.load_local(
        VECTOR_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

# ----------------------------
# Initialize Vector DB
# ----------------------------

if uploaded_files:

    with st.spinner("Processing PDFs..."):

        vector_db = create_vector_db(uploaded_files)

    st.success("PDFs indexed successfully!")

elif os.path.exists(VECTOR_PATH):

    vector_db = load_vector_db()

else:

    vector_db = None

# ----------------------------
# Chat System
# ----------------------------

if vector_db:

    retriever = vector_db.as_retriever(
        search_kwargs={"k": 4}
    )

    llm = ChatGroq(
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model_name="llama-3.1-8b-instant"
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever,
        memory=memory
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask a research question")

    if prompt:

        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        # ----------------------------
        # Decide if Web Search Needed
        # ----------------------------

        web_keywords = ["current", "latest", "today", "news", "time", "weather"]

        if any(word in prompt.lower() for word in web_keywords):

            web_result = search.invoke(prompt)

        else:

            web_result = "No web search needed."

        pdf_result = qa_chain.invoke({"question": prompt})

        answer = f"""
### 📄 PDF Knowledge Base
{pdf_result['answer']}

### 🌐 Web Results
{web_result}
"""

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

else:

    st.info("Upload PDFs to start.")