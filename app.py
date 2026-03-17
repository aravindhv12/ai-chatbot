import os
import streamlit as st
import tempfile
import time

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from duckduckgo_search import DDGS

# ----------------------------
# DISABLE TELEMETRY
# ----------------------------
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_IMPL"] = "none"

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="AI PDF + Web Chatbot", layout="wide")
st.title("🤖 AI PDF + Web Chatbot")

# ----------------------------
# CUSTOM CSS
# ----------------------------
st.markdown("""
<style>
.chat-user {
    background-color: #4CAF50;
    color: white;
    padding: 12px;
    border-radius: 12px;
    margin: 8px;
}
.chat-bot {
    background-color: #f1f0f0;
    padding: 12px;
    border-radius: 12px;
    margin: 8px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# API KEY
# ----------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ Add GROQ_API_KEY in Streamlit secrets")
    st.stop()

# ----------------------------
# LLM INIT
# ----------------------------
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"
)

# ----------------------------
# SESSION STATE INIT
# ----------------------------
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "sources" not in st.session_state:
    st.session_state.sources = []

# ----------------------------
# FILE UPLOADER
# ----------------------------
uploaded_files = st.file_uploader(
    "📄 Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    key="pdf_uploader_unique"
)

# ----------------------------
# PROCESS PDF SAFELY
# ----------------------------
if uploaded_files and st.session_state.vector_db is None:
    docs = []
    names = []

    for file in uploaded_files:
        try:
            file.seek(0)
            file_bytes = file.read()
            if not file_bytes:
                st.warning(f"⚠️ Skipping empty file: {file.name}")
                continue

            names.append(file.name)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            loaded_docs = loader.load()
            if not loaded_docs:
                st.warning(f"⚠️ No readable content in: {file.name}")
                continue
            docs.extend(loaded_docs)
        except Exception as e:
            st.error(f"❌ Error processing {file.name}: {str(e)}")

    if docs:
        st.success(f"✅ Loaded: {', '.join(names)}")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = splitter.split_documents(docs)

        embeddings = FakeEmbeddings(size=384)
        persist_dir = os.path.join(tempfile.gettempdir(), f"chroma_{time.time()}")
        st.session_state.vector_db = Chroma.from_documents(
            split_docs,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        st.success("✅ PDF processed successfully!")
    else:
        st.error("❌ No valid PDFs uploaded.")

# ----------------------------
# SIDEBAR OPTIONS
# ----------------------------
use_web = st.sidebar.checkbox("🌐 Enable Web Search")

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.sources = []
    st.session_state.vector_db = None

# ----------------------------
# DISPLAY CHAT HISTORY
# ----------------------------
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"<div class='chat-user'>🧑 {msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bot'>🤖 {msg}</div>", unsafe_allow_html=True)

# ----------------------------
# USER INPUT
# ----------------------------
query = st.chat_input("Ask something...")

# ----------------------------
# HANDLE QUERY (PDF + WEB)
# ----------------------------
if query:
    st.session_state.chat_history.append(("user", query))
    response = ""
    sources = []

    # PDF RETRIEVAL
    if st.session_state.vector_db:
        try:
            retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(query)
            for d in docs:
                sources.append(d.page_content[:200])
            qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
            pdf_answer = qa.run(query)
            response += f"📄 {pdf_answer}\n\n"
        except Exception as e:
            response += f"PDF Error: {str(e)}\n"

    # WEB SEARCH
    if use_web:
        try:
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=3):
                    results.append(f"🔹 {r['title']}\n{r['body']}")
            web_text = "\n".join(results)  # <-- Build string outside f-string
            response += f"🌐 {web_text}"
        except Exception as e:
            response += f"Web Error: {str(e)}\n"

    if not response:
        response = "⚠️ No results found."

    st.session_state.chat_history.append(("bot", response))
    st.session_state.sources = sources

# ----------------------------
# DISPLAY SOURCES
# ----------------------------
if st.session_state.sources:
    st.sidebar.markdown("### 📚 Sources")
    for i, s in enumerate(st.session_state.sources):
        st.sidebar.write(f"{i+1}. {s[:150]}...")
