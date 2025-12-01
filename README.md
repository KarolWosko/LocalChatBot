# LocalChatBot
Offline Python desktop app for analyzing PDF, TXT and DOCX files using ONNX embeddings, FAISS vector search and local LLM models (Zephyr and Mistral). Designed for fast, private and intelligent document querying.

# Features
Intelligent Document Search:
- Reads PDF, TXT, DOCX.
- Automatic chunking and embedding generation (ONNX model).
- FAISS vector database for fast semantic search.

Local Large Language Models:
- Works fully offline.
- Supports:
  - Zephyr 7B Beta (GGUF).
  - Mistral 7B (GGUF).
  - Other Llama-compatible models.
- Token counting via AutoTokenizer (Transformers).

Desktop GUI (PyQt6):
- Loading dialog window.
- File preview before loading.
- Automatic encoding detection (chardet).
- Local caching of embeddings & logs.

Architecture Highlights:
- Clean separation of:
  - file loading,
  - embeddings,
  - FAISS index,
  - model inference,
  - GUI processing.
- Full EXE build support (PyInstaller/Nuitka).

# Project Architecture
LocalChatBot/
|
|- main.py
|
|- language_model/     <- place GGUF LLM models here (ignored by Git)
|- models/             <- place ONNX embedding models here (ignored by Git)
|- faiss_data/         <- FAISS index files (ignored)
|- logs/               <- runtime logs (ignored)
|
|- requirements.txt
|- README.md
|- .gitignore

# Installation
1. Create virtual environment:
   python -m venv .venv
   .venv\Scripts\activate # on Windows
2. Install dependencies:
   pip install -r requirements.txt
3. Run the app:
   python src/main.py

# Download models
Place your .gguf models in /language_model/
Place your ONNX embedding model in /models/

---

This project is designed purely for local, offline AI workflows and does not send any data to external APIs or servers.
