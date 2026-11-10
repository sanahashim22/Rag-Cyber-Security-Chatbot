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
│   ├── dashboard_app.py            ← Main dashboard (Single-Model Ranking, 4-Model Comparison, floating chatbot)
│   ├── RAGStream.py                ← Older chat-first app (standards Q&A only)
│   ├── snort_ranking.py            ← Snort alert severity ranking + correlation pipeline
│   ├── predefined_rules.json       ← Reference-only rule rubric (not shown to the models)
│   ├── rule_docs_preprocessed_by_sid.json  ← Snort rule lookup table (plain sid → rule-doc dict,
│   │                                          NOT in docs/, NOT embedded/retrieved - a direct dict lookup)
│   ├── load_documents.py
│   ├── memory_monitor.py
│   ├── docs/                       ← PUT YOUR PDF/TXT FILES HERE (knowledge base, chunked + embedded)
│   │   ├── NIST_CSF_2.0.txt
│   │   └── ISO_IEC_27001_2022.txt
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

**Supported formats:** `.pdf` and `.txt` only - these are chunked and embedded into the RAG knowledge base.

> **Note:** `rule_docs_preprocessed_by_sid.json` (Snort rule documentation) is intentionally kept **outside** `docs/`, in `RAGStream/` itself. It's not chunked, embedded, or vector-searched - it's loaded once into memory and looked up directly by a `sid` key (a plain dict lookup, not RAG retrieval).

---

## Snort Alert Severity Ranking

The main dashboard (`dashboard_app.py`) has a **Single-Model Ranking** tab and a **4-Model Comparison** tab. Upload a Snort/Suricata alerts JSON export and click "Run ranking" to process it (no special prompt needed: it's auto-detected by checking for fields like `sid`, `priority`, or an `alerts` list).

For each alert, the app:
1. Retrieves relevant context from the knowledge base (NIST CSF 2.0, ISO/IEC 27001) via the same RAG pipeline used for chat, looks up the alert's Snort rule doc directly by `sid`, and sends up to **4000 characters** of the retrieved context to the model (actual retrieval is often ~6000 chars across 4 chunks; this cap is a deliberate token-budget tradeoff, not a bug).
2. Asks the selected LLM to rank the alert's severity 1-5 (1 = Critical, 5 = Informational/noise), citing the retrieved evidence, **not** a predefined rule rubric; the model reasons from the alert data and retrieved documents on its own.
3. Repeats this 3 times per alert to check whether the model's judgment is consistent.
4. Has a second, independent model (`gemma2:2b`, the fixed judge model) score the justification's quality (0.0-1.0), and flags any case where the rank disagrees with Snort's own priority.
5. Looks across the whole batch for **combined/correlated events** - a two-step process where the model first proposes groups of alerts that form one connected sequence (alerts must stay close together in chronological order - at most one unrelated alert between two group members), then a second, independent "skeptical reviewer" LLM call re-examines each proposed group on its own and can shrink or reject it if the alerts don't really belong together. Surviving groups are shown as `🔗 AI-suggested` combined events.

Results are shown as a formatted report (rank distribution, SID-match check, judge scores, per-alert justifications, combined-event groups) with a button to download the full report as `.txt`/Excel, plus a per-stage timing breakdown.

A floating chatbot widget (bottom-right) lets you ask standards questions at any time; it's disabled while a ranking run is in progress so it can't interrupt or cancel the run.

`predefined_rules.json` is loaded in code but is **not** shown to the ranking/judge models; it's kept only as a separate, documented reference point you can compare the models' independent answers against.

---

## Running the Main App (RAGStream)

### Requirements
- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- The following models pulled in Ollama:
  - `nomic-embed-text` (embedding)
  - `llama3.2` (ranking model, 3.2B)
  - `qwen2.5:1.5b` (ranking model, 1.5B)
  - `phi4-mini` (ranking model, 3.8B)
  - `gemma4:e2b` (ranking model, 5.1B raw / ~2B-class effective footprint)
  - `gemma2:2b` (fixed judge model - scores justification quality, not user-selectable)

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
NIST CSF 2.0 and ISO/IEC 27001:2022 are already included in `RAGStream/docs/`; the Snort rule lookup table (`rule_docs_preprocessed_by_sid.json`) is already included in `RAGStream/` itself. No setup needed.

### Run the app
```bash
cd RAGStream
streamlit run dashboard_app.py
```

Open your browser at `http://localhost:8501`

(`RAGStream.py` is the older, chat-first app - standards Q&A only, no alert ranking/correlation UI. `dashboard_app.py` is the actively maintained app and the one described above.)

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

| Model | Parameters | Use |
|---|---|---|
| `llama3.2` | 3.2B | Default ranking LLM |
| `qwen2.5:1.5b` | 1.5B | Lightweight ranking option |
| `phi4-mini` | 3.8B | Microsoft Phi ranking option |
| `gemma4:e2b` | 5.1B raw (~2B-class effective) | Google Gemma ranking option |
| `gemma2:2b` | 2B | Fixed justification-quality judge model (not user-selectable) |
| `nomic-embed-text` | - | Embeddings (always used) |

The four ranking models (`RANKING_MODELS` in `dashboard_app.py`) can be changed in code; the judge model (`JUDGE_MODEL` in `snort_ranking.py`) is fixed.

---

## Author

**Sana Hashim**

MIT License — see [LICENSE](LICENSE) for details.
