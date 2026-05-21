# RAG Demo — Document Q&A with LangChain & Groq

A Retrieval-Augmented Generation (RAG) pipeline that answers questions 
about any PDF document using LangChain, FAISS, and a free Groq LLM.

## How it works

1. **Ingest** — PDF is loaded and split into 500-token chunks
2. **Embed** — chunks are embedded using sentence-transformers 
   (all-MiniLM-L6-v2) and stored in a FAISS vector store
3. **Retrieve** — top-3 most relevant chunks are retrieved per question
4. **Generate** — Groq's LLaMA 3 generates an answer grounded 
   in the retrieved context

## Stack
- LangChain — pipeline orchestration
- FAISS — vector similarity search
- Hugging Face sentence-transformers — embeddings (free, local)
- Groq API — LLM inference (free tier)

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # add your Groq API key
python main.py
```

## Why I built this

Built as part of my preparation for research work in LLM-based 
AI agent systems. RAG is a foundational 
pattern for building reliable, grounded AI agents.