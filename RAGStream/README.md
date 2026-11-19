# RAGStream - Cybersecurity Standards RAG + Snort/Suricata Alert Ranking

**RAGStream** is the main Streamlit app for this project. It does two things:

1. **Standards Q&A**: ask questions about **NIST Cybersecurity Framework (CSF) 2.0** and **ISO/IEC 27001:2022**, answered via RAG retrieval over the actual standard documents.
2. **Snort/Suricata alert severity ranking**: rank IDS alerts 1-5 for severity (1 = critical, 5 = informational/noise), 3 times per alert to check consistency, with an independent model scoring the justification's quality against the retrieved evidence.

Everything runs locally via **Ollama** - no cloud API calls, fully offline and private.

> This file covers running the app from inside this `RAGStream/` folder. See the [repo root README](../README.md) for the full project structure and the other (command-line) RAG demos.

## Running it

```bash
cd RAGStream
pip install -r requirements.txt
pip install pdfplumber  # for loading PDF standards documents into docs/
streamlit run dashboard_app.py
```

Open `http://localhost:8501`. Requires [Ollama](https://ollama.com) installed and running locally, with these models pulled:

- `nomic-embed-text` (embeddings)
- `llama3.2`, `qwen2.5:1.5b`, `phi4-mini`, `gemma4:e2b` (the 4 selectable ranking models)
- `gemma2:2b` (fixed judge model)

By default this uses your local CPU Ollama. For a remote/GPU Ollama instead (e.g. exposed via `ngrok http 11434`), use `run_gpu_test.ps1` rather than setting `OLLAMA_HOST` yourself:

```powershell
.\run_gpu_test.ps1 -OllamaHost "https://xxxx.ngrok-free.dev"
```

(This app reads `RAG_OLLAMA_HOST`, not Ollama's own `OLLAMA_HOST` - deliberately a different name, so a system-wide `OLLAMA_HOST` set for Ollama's *server* bind address never gets mistaken for where this app's *client* should connect.)

(`RAGStream.py` is the older, chat-first app - standards Q&A only, no alert ranking UI. `dashboard_app.py` is the actively maintained one, described below.)

## Getting alerts into the dashboard

- **Drop a file** into `alert_inputs/` - detected live, every few seconds, without restarting Streamlit. Accepts this project's own flat alerts-list format, and IDS-Dashboard's `ids.combined-detections` export format (rule definitions + a packet timeline) - both get normalized into the same shape before ranking.
- **Receive pushes**: `python alert_receiver.py` listens for alerts pushed from an [IDS-Dashboard](https://github.com/Moeed0/IDS-Dashboard) sensor's own "Send" button and saves them into `alert_inputs/` automatically.
- **Pull on demand**: the dashboard's "📥 Receive from IDS-Dashboard" panel pulls alerts directly from a sensor's export API - by time window (5 minutes - 7 days) or by "Last N alerts".
- A dropdown lets you rank one specific file in `alert_inputs/`, or all of them combined. More than 10 alerts combined triggers a confirmation step for how many to actually process.

## Rule documentation & threat intel

Each alert's rule documentation is looked up by `sid`, trying two sources in order:

1. **`intel.db`** - a shared SQLite database (Talos/Snort rule docs, CVEs with CVSS/EPSS/CISA-KEV, MITRE ATT&CK) downloaded from IDS-Dashboard via the "🗄️ Threat-intel database" panel. ~1.4GB, not auto-refreshed (too big to poll live) - use that panel's "Check for updates" / "Download / update" buttons.
2. **`rule_docs_preprocessed_by_sid.json`** - a local, pre-built table of this project's own Snort ruleset, used as a fallback when intel.db has nothing for that sid (or hasn't been downloaded).

Neither is chunked/embedded/vector-searched - both are exact `sid` dict lookups, independent of the RAG retrieval used for the NIST/ISO context.

Every table, the Excel export, and the downloadable report show a **Source** column (Snort/Suricata) and an **Actual Ranking** column - the alert's own reported severity. IDS-Dashboard names its own severity in words (Low/Medium/High/Critical); these are shown as e.g. `Critical - 1`, `Low - 4`, using this project's own numbering (1 = most severe) rather than passing the other system's number through as-is.

## Files in this folder

| File | Purpose |
|---|---|
| `dashboard_app.py` | Main dashboard - Single-Model Ranking, 4-Model Comparison, floating chatbot |
| `RAGStream.py` | Older chat-first app (standards Q&A only) |
| `snort_ranking.py` | Alert severity ranking + correlation pipeline |
| `intel_lookup.py` | Copy of IDS-Dashboard's own script - reads the shared `intel.db` |
| `alert_receiver.py` | Standalone listener for IDS-Dashboard's "Send" button |
| `alert_inputs/` | Drop alert JSON files here - not committed to git |
| `intel.db` | Downloaded on demand (~1.4GB) - not committed to git |
| `rule_docs_preprocessed_by_sid.json` | Fallback rule-doc lookup table (this project's own ruleset) |
| `predefined_rules.json` | Reference-only rule rubric - **not** shown to the ranking/judge models |
| `load_documents.py`, `memory_monitor.py` | Supporting modules |
| `docs/` | NIST/ISO (and any other) standards documents - chunked + embedded for RAG |
| `run_gpu_test.ps1` | Launches the app pointed at a remote/GPU Ollama instead of local CPU |

## Model selection

The 4 ranking models (`RANKING_MODELS` in `dashboard_app.py`) and the fixed judge model (`JUDGE_MODEL` in `snort_ranking.py`) can be changed in code.

## License

MIT License - see the [repo root](../LICENSE).
