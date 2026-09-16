"""
judge_bertscore.py
-------------------
Stage 6: BERTScore -- token-level contextual embedding similarity between
ground truth and model response, using precision/recall/F1 from BERT
embeddings (via the bert-score package, DeBERTa backbone by default).

Unlike SBERT/cosine (which compare ONE embedding per whole text), BERTScore
aligns individual tokens between the two texts and is more sensitive to
whether specific facts/entities are actually present -- good for catching
cases where a response is topically similar but misses the key detail.

SETUP:
    pip install bert-score --break-system-packages   (if on this machine's venv, drop the flag)

USAGE:
    python judge_bertscore.py

OUTPUT:
    Adds "bertscore_precision", "bertscore_recall", "bertscore_f1" (floats)
    and "bertscore_verdict" (1/0) to each response inside raw_responses.json.

NOTE: BERTScore's F1 values sit in a narrower, higher range than cosine
similarity (typically 0.85-0.95 even for so-so answers), so its threshold
is deliberately higher than the SBERT/cosine ones -- see README_EVAL.md.
"""

import json
import os

from bert_score import score as bert_score

INPUT_FILE = os.path.join(os.path.dirname(__file__), "raw_responses.json")
THRESHOLD = 0.85  # BERTScore F1 sits higher than cosine similarity -- tune after eyeballing


def get_response_text(response):
    return response if isinstance(response, str) else response["text"]


def run():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Collect everything first so we can batch-score with bert-score (much faster
    # than calling it once per pair -- it loads a full transformer model).
    keys = []  # (qid, model_name)
    candidates = []
    references = []

    for qid, item in results.items():
        ground_truth = item["ground_truth"]
        for model_name, response in item["responses"].items():
            if isinstance(response, dict) and "bertscore_f1" in response:
                continue
            keys.append((qid, model_name))
            candidates.append(get_response_text(response))
            references.append(ground_truth)

    if not keys:
        print("Nothing new to score -- all responses already have bertscore_f1.")
        return

    print(f"Scoring {len(keys)} response(s) with BERTScore (downloads a model on first run)...")
    P, R, F1 = bert_score(candidates, references, lang="en", verbose=True)

    for (qid, model_name), p, r, f1 in zip(keys, P.tolist(), R.tolist(), F1.tolist()):
        response = results[qid]["responses"][model_name]
        text = get_response_text(response)
        verdict = int(f1 >= THRESHOLD)
        results[qid]["responses"][model_name] = {
            "text": text,
            **(response if isinstance(response, dict) else {}),
            "bertscore_precision": p,
            "bertscore_recall": r,
            "bertscore_f1": f1,
            "bertscore_verdict": verdict,
        }

    with open(INPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Done. bertscore_precision/recall/f1/verdict added to raw_responses.json")


if __name__ == "__main__":
    run()
