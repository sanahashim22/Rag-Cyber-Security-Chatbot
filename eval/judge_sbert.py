"""
judge_sbert.py
--------------
Stage 4: Sentence-BERT (SBERT) cosine similarity, using all-MiniLM-L6-v2
from sentence-transformers.

Why this is a useful second opinion alongside nomic-embed-text: SBERT's
MiniLM model is trained specifically for sentence-level semantic similarity
(it's the standard STS-benchmark model), whereas nomic-embed-text is a
general-purpose retrieval embedding model. Comparing both tells you whether
your "correct" verdicts are robust across embedding models, or just an
artifact of one model's quirks.

USAGE:
    python judge_sbert.py

OUTPUT:
    Adds "sbert_score" (float) and "sbert_verdict" (1/0) to each response
    inside raw_responses.json.
"""

import json
import os

from sentence_transformers import SentenceTransformer, util

INPUT_FILE = os.path.join(os.path.dirname(__file__), "raw_responses.json")
MODEL_NAME = "all-MiniLM-L6-v2"
THRESHOLD = 0.75  # tune this after eyeballing a few scores -- see README_EVAL.md


def get_response_text(response):
    return response if isinstance(response, str) else response["text"]


def run():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

    print(f"Loading {MODEL_NAME} (first run downloads it, ~90MB)...")
    model = SentenceTransformer(MODEL_NAME)

    for qid, item in results.items():
        ground_truth = item["ground_truth"]
        gt_embedding = model.encode(ground_truth, convert_to_tensor=True)

        for model_name, response in item["responses"].items():
            if isinstance(response, dict) and "sbert_score" in response:
                continue

            text = get_response_text(response)
            print(f"[{qid}] sbert scoring {model_name}...")

            resp_embedding = model.encode(text, convert_to_tensor=True)
            score = float(util.cos_sim(gt_embedding, resp_embedding).item())
            verdict = int(score >= THRESHOLD)

            item["responses"][model_name] = {
                "text": text,
                **(response if isinstance(response, dict) else {}),
                "sbert_score": score,
                "sbert_verdict": verdict,
            }

            with open(INPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    print("Done. sbert_score / verdict added to raw_responses.json")


if __name__ == "__main__":
    run()
