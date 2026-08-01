import streamlit as st
import ollama
import PyPDF2
import io
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="centered")

st.title("📄 AI Document Q&A")
st.caption("Upload a PDF and ask questions — powered by Llama 3.2 + RAG")

# ─── Session State ─────────────────────────────────────────────
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

# ─── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        if uploaded_file.name != st.session_state.doc_name:
            with st.spinner("Reading and indexing document..."):
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )
                chunks = splitter.split_text(text)

                embeddings = OllamaEmbeddings(model="nomic-embed-text")
                vectorstore = Chroma.from_texts(
                    texts=chunks,
                    embedding=embeddings,
                    persist_directory="./chroma_db"
                )

                st.session_state.vectorstore = vectorstore
                st.session_state.doc_name = uploaded_file.name
                st.session_state.messages = []

            st.success(f"✅ '{uploaded_file.name}' indexed successfully!")
            st.info(f"📊 {len(chunks)} chunks created")

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    if st.session_state.doc_name:
        st.markdown(f"**Active Document:**\n{st.session_state.doc_name}")

# ─── Chat History ──────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ─── Chat Input
if st.session_state.vectorstore is None:
    st.info("👆 Please upload a PDF from the sidebar to get started.")

prompt = st.chat_input("Ask a question about your document...")

if prompt:
    if st.session_state.vectorstore is None:
        st.warning("⚠️ Please upload a PDF document first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching document and generating answer..."):
                try:
                    docs = st.session_state.vectorstore.similarity_search(prompt, k=3)
                    context = "\n\n".join([doc.page_content for doc in docs])

                    full_prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided document context below.
If the answer is not in the context, say "I couldn't find that in the document."

Context from document:
{context}

Question: {prompt}

Answer:"""

                    response = ollama.chat(
                        model="llama3.2:1b",
                        messages=[{"role": "user", "content": full_prompt}]
                    )

                    reply = response['message']['content']
                    st.markdown(reply)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": reply
                    })

                    with st.expander("📚 Source chunks used"):
                        for i, doc in enumerate(docs):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.markdown(doc.page_content)
                            st.divider()

                except Exception as e:
                    st.error(f"Error: {str(e)}")