"""
combine_results.py
--------------------
Stage 7-8: Combine raw_responses.json (which now carries scores from all
4 judges) into one flat evaluation_results.csv, and print/save a final
accuracy-per-model-per-metric comparison table.

USAGE:
    python combine_results.py

OUTPUT:
    evaluation_results.csv    -- one row per (query, model)
    accuracy_summary.csv      -- one row per model, one column per metric
"""

import json
import os

import pandas as pd

INPUT_FILE = os.path.join(os.path.dirname(__file__), "raw_responses.json")
RESULTS_CSV = os.path.join(os.path.dirname(__file__), "evaluation_results.csv")
SUMMARY_CSV = os.path.join(os.path.dirname(__file__), "accuracy_summary.csv")

METRIC_VERDICT_COLUMNS = {
    "LLM-as-judge (Groq)": "groq_verdict",
    "Embedding cosine (nomic-embed-text)": "embedding_cosine_verdict",
    "SBERT (all-MiniLM-L6-v2)": "sbert_verdict",
    "BERTScore": "bertscore_verdict",
}


def run():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

    rows = []
    for qid, item in results.items():
        query, ground_truth = item["query"], item["ground_truth"]
        for model_name, response in item["responses"].items():
            if not isinstance(response, dict):
                # Not judged by anything yet -- skip, run the judge_*.py scripts first.
                continue
            rows.append({
                "query_id": qid,
                "query": query,
                "ground_truth": ground_truth,
                "model": model_name,
                "model_response": response.get("text", ""),
                "groq_verdict": response.get("groq_verdict"),
                "groq_raw": response.get("groq_raw"),
                "embedding_cosine_score": response.get("embedding_cosine_score"),
                "embedding_cosine_verdict": response.get("embedding_cosine_verdict"),
                "sbert_score": response.get("sbert_score"),
                "sbert_verdict": response.get("sbert_verdict"),
                "bertscore_precision": response.get("bertscore_precision"),
                "bertscore_recall": response.get("bertscore_recall"),
                "bertscore_f1": response.get("bertscore_f1"),
                "bertscore_verdict": response.get("bertscore_verdict"),
            })

    if not rows:
        print("No judged responses found. Run get_responses.py then the judge_*.py scripts first.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"Saved {len(df)} rows to {RESULTS_CSV}")

    # Stage 8: accuracy % per model per metric
    summary_rows = []
    for model_name, group in df.groupby("model"):
        row = {"model": model_name, "num_queries": len(group)}
        for metric_label, col in METRIC_VERDICT_COLUMNS.items():
            valid = group[col].dropna()
            accuracy = (valid.sum() / len(valid) * 100) if len(valid) else None
            row[metric_label] = round(accuracy, 1) if accuracy is not None else "not scored yet"
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    print(f"\nSaved accuracy summary to {SUMMARY_CSV}\n")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    run()
