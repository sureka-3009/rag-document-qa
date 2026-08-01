# 📄 AI Document Q&A (RAG Pipeline)

A RAG-based Document Q&A system built with Python, Streamlit, LangChain, 
ChromaDB, and Llama 3.2 — upload any PDF and ask questions in plain English, 
powered entirely by local AI with zero API cost.

## Features
- Upload any PDF and ask questions in natural language
- Answers grounded only in your document — no hallucinations
- Semantic search using vector embeddings (ChromaDB)
- Source chunk transparency — see exactly what the AI referenced
- 100% local — no API key, no cloud, fully private
- Real-time Streamlit interface with chat history

## Tech Stack
Python | Streamlit | LangChain | ChromaDB | Ollama | Llama 3.2 | nomic-embed-text | RAG Pipeline

## Setup
1. Install Ollama from ollama.com
2. Run: `ollama pull llama3.2:1b`
3. Run: `ollama pull nomic-embed-text`
4. Clone this repo
5. `pip install -r requirements.txt`
6. `streamlit run app.py`

## How RAG Works
The LLM itself has no knowledge of your document. When you upload a PDF, 
it gets split into chunks and stored as vector embeddings in ChromaDB. 
When you ask a question, the system finds the most semantically similar 
chunks and sends them as context to Llama 3.2 — so the answer always 
comes from your actual document, not general knowledge.