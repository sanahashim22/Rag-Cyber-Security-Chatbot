# 🔒 Cybersecurity Standards RAG System

An AI-powered question-answering system built for cybersecurity standards and frameworks. Currently supports **NIST Cybersecurity Framework (CSF) 2.0** and is designed to easily accommodate additional standards as they are added.

Built with **Streamlit**, **Ollama**, and local LLMs — fully offline, private, and free to run.

---

## Project Structure

```
RAG-with-5-models/
├── RAGStream/          ← Main Streamlit web app (start here)
│   ├── RAGStream.py
│   ├── load_documents.py
│   ├── memory_monitor.py
│   ├── docs/           ← PUT YOUR PDF/TXT FILES HERE
│   │   └── HOW_TO_ADD_CONTENT.txt
│   └── pyproject.toml
├── ConvoRAG/           ← Conversational RAG (command-line)
├── SimpleRAG/          ← Basic single-turn RAG (command-line)
├── SemanticSeek/       ← Semantic search only (command-line)
├── CosineExplorer/     ← Cosine similarity visualizer (command-line)
└── README.md
```

---

## How to Add New Cybersecurity Content

> This is the most important thing to know. Follow these simple steps.

### Step 1 — Get your document
Your instructor gives you a new PDF or TXT file (e.g., `ISO_27001.pdf`)

### Step 2 — Copy it to the docs folder
```
RAGStream/docs/ISO_27001.pdf
```
Just drag and drop the file into that folder.

### Step 3 — Restart the app
In the browser, click **"Restart & Reload Documents"** in the sidebar.

### Step 4 — Done!
The system will automatically detect and load all files in the `docs/` folder.
You can have **multiple documents at the same time** — all are combined into one knowledge base.

**Supported formats:** `.pdf` and `.txt`

---

## Running the Main App (RAGStream)

### Requirements
- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- The following models pulled in Ollama:
  - `nomic-embed-text` (embedding)
  - `llama3.2` (LLM)
  - `qwen2.5:1.5b`
  - `gemma2:2b`
  - `phi4-mini`
  - `gemma4:e2b`

### Install dependencies
```bash
cd RAGStream
pip install streamlit ollama numpy pdfplumber
```

### (Optional) Install Advanced Dependencies for Hybrid Search
For faster startups and **Hybrid Search** (combining vector search with BM25 keyword matching for exact IDs), install these two additional packages. If you skip this, the app will automatically fall back to basic in-memory mode!
```bash
pip install chromadb rank_bm25
```

Or with Poetry:
```bash
cd RAGStream
poetry install
poetry add pdfplumber
```

### Place your PDF in the docs folder
```bash
# Copy your NIST CSF 2.0 PDF into:
RAGStream/docs/NIST.CSWP.29.pdf
```

### Run the app
```bash
cd RAGStream
streamlit run RAGStream.py
```

Open your browser at `http://localhost:8501`

---

## Running the Command-Line Tools

These are simpler tools for learning how RAG works step by step.

```bash
# Semantic Search only (no LLM)
cd SemanticSeek
python SemanticSeek.py

# Basic RAG (single-turn Q&A)
cd SimpleRAG
python SimpleRAG.py

# Conversational RAG (multi-turn Q&A)
cd ConvoRAG
python ConvoRAG.py

# Cosine Similarity Visualizer (2D math demo)
cd CosineExplorer
python CosineExplorer.py
```

---

## Supported Cybersecurity Standards

| Standard | Status |
|---|---|
| NIST CSF 2.0 | ✅ Built-in + PDF upload |
| ISO 27001 | 📥 Add PDF to docs/ folder |
| NIST SP 800-53 | 📥 Add PDF to docs/ folder |
| Any other standard | 📥 Add PDF or TXT to docs/ folder |

---

## Models Supported

| Model | Use |
|---|---|
| `llama3.2` | Default LLM |
| `qwen2.5:1.5b` | Lightweight option |
| `gemma2:2b` | Google Gemma |
| `phi4-mini` | Microsoft Phi |
| `gemma4:e2b` | Gemma 4 |
| `nomic-embed-text` | Embeddings (always used) |

You can add or remove models from the dropdown in `RAGStream.py`.

---

## Author

**Sana Hashim**

MIT License — see [LICENSE](LICENSE) for details.
