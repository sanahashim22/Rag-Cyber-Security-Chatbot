# 🔒 Cybersecurity Standards RAG System

An AI-powered assistant for cybersecurity standards and frameworks that also ranks Snort IDS alert severity. It does two things:

1. **Standards Q&A**: ask questions about **NIST Cybersecurity Framework (CSF) 2.0** and **ISO/IEC 27001:2022**, with answers grounded in the actual standard documents via RAG retrieval. Designed to easily accommodate additional standards as they're added.
2. **Snort alert severity ranking**: attach a Snort/Suricata alerts JSON export in the chat (the 📎 attach icon), and each alert gets ranked 1-5 for severity, run 3 times per alert to check consistency, with an independent model scoring the justification quality.

Built with **Streamlit**, **Ollama**, and local LLMs — fully offline, private, and free to run.

---

## Project Structure

```
RAG-with-5-models/
├── RAGStream/                      ← Main Streamlit web app (start here)
│   ├── RAGStream.py
│   ├── snort_ranking.py            ← Snort alert severity ranking pipeline
│   ├── predefined_rules.json       ← Reference-only rule rubric (not shown to the models)
│   ├── load_documents.py
│   ├── memory_monitor.py
│   ├── docs/                       ← PUT YOUR PDF/TXT/JSON FILES HERE (knowledge base)
│   │   ├── NIST.CSWP.29.pdf
│   │   ├── ISO_IEC-27001-2022.pdf
│   │   ├── rule_docs_preprocessed_by_sid.json   ← Snort rule documentation, embedded for retrieval
│   │   └── HOW_TO_ADD_CONTENT.txt
│   └── pyproject.toml
├── data/                           ← Sample Snort alerts export + model comparison reports
├── ConvoRAG/                       ← Conversational RAG (command-line)
├── SimpleRAG/                      ← Basic single-turn RAG (command-line)
├── SemanticSeek/                   ← Semantic search only (command-line)
├── CosineExplorer/                 ← Cosine similarity visualizer (command-line)
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

**Supported formats:** `.pdf`, `.txt`, and `.json` (a sid-keyed dict of Snort rule docs, like `rule_docs_preprocessed_by_sid.json`, is split into one retrievable block per rule).

---

## Snort Alert Severity Ranking

Attach a Snort/Suricata alerts JSON export using the 📎 attach icon in the chat box (no special prompt needed: it's auto-detected by checking for fields like `sid`, `priority`, or an `alerts` list). Files with 10 or fewer alerts are processed automatically; larger files ask you to confirm how many to process first.

For each alert, the app:
1. Retrieves relevant context from the knowledge base (NIST CSF 2.0, ISO/IEC 27001, and the embedded Snort rule docs) via the same RAG pipeline used for chat.
2. Asks the selected LLM to rank the alert's severity 1-5 (1 = Critical, 5 = Informational/noise), citing the retrieved evidence, **not** a predefined rule rubric; the model reasons from the alert data and retrieved documents on its own.
3. Repeats this 3 times per alert to check whether the model's judgment is consistent.
4. Has a second, independent model score the justification's quality (0.0-1.0), and flags any case where the rank disagrees with Snort's own priority.

Results are shown in the chat as a formatted report (rank distribution, SID-match check, judge scores, per-alert justifications) with a button to download the full report as `.txt`.

`predefined_rules.json` is loaded in code but is **not** shown to the ranking/judge models; it's kept only as a separate, documented reference point you can compare the models' independent answers against.

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

### Knowledge base documents
NIST CSF 2.0, ISO/IEC 27001:2022, and the Snort rule docs are already included in `RAGStream/docs/`: no setup needed.

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
| NIST CSF 2.0 | ✅ Built-in |
| ISO/IEC 27001:2022 | ✅ Built-in |
| NIST SP 800-53 | 📥 Add PDF to docs/ folder |
| Any other standard | 📥 Add PDF, TXT, or JSON to docs/ folder |

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
