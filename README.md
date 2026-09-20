# AI Medical Chatbot

A Retrieval-Augmented Generation (RAG) based medical chatbot that answers questions using information retrieved from a provided medical document.

## Features

- PDF document ingestion
- Text extraction and chunking
- Chunk overlap for better context preservation
- Local embeddings using Hugging Face Sentence Transformers
- Semantic search using ChromaDB
- Answer generation using Google Gemini
- Flask backend
- Simple web-based chatbot interface
- Context-grounded responses to reduce hallucination

## Tech Stack

- Python
- Flask
- LangChain
- Google Gemini
- Hugging Face Sentence Transformers
- ChromaDB
- PyPDF
- HTML
- CSS
- JavaScript

## How It Works

```text
Medical PDF
     ↓
Text Extraction
     ↓
Text Chunking
     ↓
Hugging Face Embeddings
     ↓
ChromaDB
     ↓
User Question
     ↓
Question Embedding
     ↓
Similarity Search
     ↓
Relevant Chunks
     ↓
Google Gemini
     ↓
Final Answer

