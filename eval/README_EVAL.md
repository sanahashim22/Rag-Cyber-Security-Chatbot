# RAG Evaluation Pipeline — 4-Metric Comparison

Compares 5 Ollama models (`llama3.2`, `qwen2.5:1.5b`, `gemma2:2b`, `phi4-mini`, `gemma4:e2b`)
on the same NIST CSF 2.0 questions, scored by **4 different metrics**:

1. **Embedding cosine similarity** (`nomic-embed-text`, same model your app already uses)
2. **BERTScore** (token-level contextual similarity)
3. **Sentence-BERT / SBERT** (`all-MiniLM-L6-v2`)
4. **LLM-as-judge** (Groq API, Yes/No verdict)

## 0. Where these files go

Drop this `eval/` folder inside your **copied** project (not the original), next to `RAGStream/`:

```
latest-rag-version-2.0-eval/
  RAGStream/
    RAGStream.py
    load_documents.py
    docs/NIST.CSWP.29.pdf
  eval/                      <- these 8 files go here
    rag_core.py
    queries.py
    get_responses.py
    judge_groq.py
    judge_embedding_cosine.py
    judge_sbert.py
    judge_bertscore.py
    combine_results.py
```

## 1. Install anything missing

You've already got most of this, but to be safe (run inside your activated `venv`):

```powershell
pip install ollama numpy chromadb rank_bm25 pdfplumber
pip install sentence-transformers groq pandas bert-score
```

(`ragas`, `datasets`, `langchain-groq` are no longer needed since RAGAS was dropped —
fine to leave installed, no harm.)

## 2. Set your Groq API key (for Stage 3, LLM-as-judge)

```powershell
$env:GROQ_API_KEY="your-key-here"
```

## 3. Run the pipeline, in order

From inside `eval/`:

```powershell
cd eval

# Stage 2 — get responses. ALWAYS test first, exactly like you planned:
python get_responses.py --test          # 1 query, 1 model — sanity check
python get_responses.py                  # full run: 15 queries x 5 models = 75 calls

# Stages 3-6 — run all 4 judges (any order, each is independent):
python judge_groq.py
python judge_embedding_cosine.py
python judge_sbert.py
python judge_bertscore.py

# Stages 7-8 — combine + compute accuracy:
python combine_results.py
```

Every script is **resumable** — it saves after each response/score, and skips
anything already done. If a script crashes partway (e.g. Ollama times out on
`phi4-mini`), just rerun it.

## 4. Outputs

- `raw_responses.json` — full detail, all 4 scores per (query, model)
- `evaluation_results.csv` — flat table: query, ground_truth, model, response, all 4 verdicts/scores
- `accuracy_summary.csv` — final comparison table: 5 models × 4 metrics, accuracy %

## 5. About the thresholds

Cosine-similarity-style metrics (embedding cosine, SBERT) are set to **0.75**.
BERTScore is set to **0.85**, because BERTScore F1 sits in a naturally higher,
narrower range than raw cosine similarity — a mediocre answer can still score
~0.85-0.88 there. **Don't trust these numbers blindly.** After your first full
run, open `evaluation_results.csv`, sort by each score column, and manually
check where "clearly correct" answers stop and "clearly wrong" ones start.
Adjust `THRESHOLD` at the top of the relevant `judge_*.py` file and rerun
`combine_results.py` (no need to re-run the judges — just re-run combine
after editing verdicts, or delete the `_verdict` fields and rerun the judge
if you changed its threshold).

## 6. Why 4 metrics instead of 1

This is the direct answer to "which metric did you choose, and why not just
LLM-as-judge": each metric catches different failure modes.

- **LLM-as-judge** understands meaning best but is itself an LLM — it can be
  inconsistent, and depends on your prompt/judge model choice.
- **Embedding cosine** (nomic-embed-text) is cheap, deterministic, matches
  the embedding model already used for retrieval in the app — good baseline.
- **SBERT** is trained specifically for sentence-similarity benchmarks, so it
  acts as a second, independent semantic-similarity opinion.
- **BERTScore** works at the token level, so it's more sensitive to whether a
  specific fact/entity is actually present, not just "topically similar."

Reporting all 4 side by side (rather than picking one) is what your
instructor's feedback ("explore other options," "don't just use LLM-as-judge
alone") was asking for.
