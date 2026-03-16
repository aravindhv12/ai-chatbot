import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.tools import DuckDuckGoSearchResults


# GROQ MODEL
llm = ChatGroq(
    model="llama3-70b-8192",
    api_key=st.secrets["GROQ_API_KEY"]
)

# EMBEDDINGS
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# LOAD PDF
loader = PyPDFLoader("data/sample.pdf")
docs = loader.load()

# SPLIT TEXT
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

documents = splitter.split_documents(docs)

# VECTOR DATABASE
vector_db = Chroma.from_documents(
    documents,
    embeddings,
    persist_directory="vector_store"
)

retriever = vector_db.as_retriever()

# PROMPT
prompt = ChatPromptTemplate.from_template(
"""
You are an AI assistant.

Answer the question using the provided context.

Context:
{context}

Question:
{question}
"""
)

# WEB SEARCH
search = DuckDuckGoSearchResults()

# STREAMLIT UI
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
        st.write(response)        chunk_overlap=200
    )

    documents = splitter.split_documents(docs)

    db = Chroma.from_documents(
        documents,
        embeddings,
        persist_directory="vector_store"
    )

    return db


vector_db = load_vector_db()


# RETRIEVER
retriever = vector_db.as_retriever()


# PROMPT
prompt = ChatPromptTemplate.from_template(
"""
Answer the question using the provided context.

<context>
{context}
</context>

Question: {input}
"""
)

# DOCUMENT CHAIN
document_chain = create_stuff_documents_chain(llm, prompt)

# RAG CHAIN
rag_chain = create_retrieval_chain(retriever, document_chain)


# WEB SEARCH
search = DuckDuckGoSearchResults()


# STREAMLIT UI
st.title("AI Research Assistant")

query = st.text_input("Ask a question")

if query:

    if "search" in query.lower():

        result = search.invoke(query)

        st.write("🌐 Web Search Result:")
        st.write(result)

    else:

        response = rag_chain.invoke({"input": query})

        st.write("📄 Answer:")
        st.write(response["answer"])  


llm = load_llm()


# ---------- EMBEDDINGS ----------

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


embeddings = load_embeddings()


# ---------- VECTOR STORE ----------

VECTOR_PATH = "vector_store"


def create_vector_store(pdf):
    loader = PyPDFLoader(pdf)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    vector_db = FAISS.from_documents(
        chunks,
        embeddings
    )

    vector_db.save_local(VECTOR_PATH)

    return vector_db


@st.cache_resource
def load_vector_store():
    if os.path.exists(VECTOR_PATH):
        return FAISS.load_local(
            VECTOR_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    return None


vector_db = load_vector_store()


# ---------- PROMPT ----------

prompt = ChatPromptTemplate.from_template(
"""
Answer the question using the provided context.

<context>
{context}
</context>

Question: {input}

Give a clear helpful answer.
"""
)


# ---------- WEB SEARCH ----------

search = DuckDuckGoSearchResults()


# ---------- UI ----------

st.sidebar.header("Upload PDF")

pdf = st.sidebar.file_uploader(
    "Upload a PDF",
    type="pdf"
)

if pdf:
    with open("temp.pdf", "wb") as f:
        f.write(pdf.getbuffer())

    vector_db = create_vector_store("temp.pdf")

    st.sidebar.success("PDF processed!")


# ---------- CHAT ----------

user_question = st.text_input("Ask a question")


if user_question:

    llm = load_llm()

    # ---------- WEB SEARCH MODE ----------

    if "search" in user_question.lower() or "current" in user_question.lower():

        web_result = search.invoke(user_question)

        response = llm.invoke(
            f"""
            Answer the question using this web result.

            {web_result}

            Question: {user_question}
            """
        )

        st.write(response.content)

    # ---------- PDF SEARCH MODE ----------

    elif vector_db:

        retriever = vector_db.as_retriever()

        document_chain = create_stuff_documents_chain(
            llm,
            prompt
        )

        retrieval_chain = create_retrieval_chain(
            retriever,
            document_chain
        )

        result = retrieval_chain.invoke(
            {"input": user_question}
        )

        st.write(result["answer"])

    else:

        response = llm.invoke(user_question)

        st.write(response.content)
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
