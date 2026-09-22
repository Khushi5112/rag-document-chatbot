# RAG-based Document Question Answering Chatbot

An interactive chatbot that lets you upload any document (PDF/text) and ask questions about it in plain English — answers are grounded in the document itself using Retrieval-Augmented Generation (RAG), reducing hallucinated or irrelevant answers.

## Overview
- Splits documents into chunks and converts them into **embeddings** (Google Gemini)
- Stores and searches embeddings using **FAISS** for fast semantic search
- Retrieves the most relevant chunks and passes them to an **LLM (Gemini)** to generate grounded answers
- Shows exactly which parts of the document were used to answer (source transparency)
- Deployed as an interactive **Streamlit** chat app

## Tech Stack
Python, LangChain, FAISS, Google Gemini API, Streamlit

## Files
- `app.py` — Streamlit chat app (upload a document, chat with it in real time)
- `RAG_Document_QA_Chatbot.ipynb` — step-by-step notebook showing how the RAG pipeline is built

## How to run
```bash
pip install streamlit langchain langchain-community langchain-google-genai langchain-text-splitters faiss-cpu pypdf
streamlit run app.py
```
You'll need a free Gemini API key from [aistudio.google.com](https://aistudio.google.com).
