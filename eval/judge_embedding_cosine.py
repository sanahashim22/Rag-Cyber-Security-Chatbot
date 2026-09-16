"""
judge_embedding_cosine.py
--------------------------
Stage 5: Cosine similarity of embeddings, using the SAME embedding model
(nomic-embed-text via Ollama) and same math as CosineExplorer.py / RAGStream.py.

For each (ground_truth, model_response) pair:
    1. Embed both texts with nomic-embed-text
    2. Compute cosine similarity
    3. Verdict = correct if similarity >= THRESHOLD

USAGE:
    python judge_embedding_cosine.py

OUTPUT:
    Adds "embedding_cosine_score" (float) and "embedding_cosine_verdict" (1/0)
    to each response inside raw_responses.json.
"""

import json
import os

import numpy as np
import ollama

INPUT_FILE = os.path.join(os.path.dirname(__file__), "raw_responses.json")
EMBEDDING_MODEL = "nomic-embed-text"
THRESHOLD = 0.75  # tune this after eyeballing a few scores -- see README_EVAL.md


def get_response_text(response):
    return response if isinstance(response, str) else response["text"]


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    dot = np.dot(v1, v2)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(dot / (n1 * n2))


def embed(text: str) -> np.ndarray:
    resp = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
    return np.array(resp["embedding"])


def run():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

    for qid, item in results.items():
        ground_truth = item["ground_truth"]
        gt_embedding = embed(ground_truth)

        for model, response in item["responses"].items():
            if isinstance(response, dict) and "embedding_cosine_score" in response:
                continue

            text = get_response_text(response)
            print(f"[{qid}] embedding-cosine scoring {model}...")
            try:
                resp_embedding = embed(text)
                score = cosine_similarity(gt_embedding, resp_embedding)
            except Exception as e:
                print(f"    error: {e}")
                score = None

            verdict = int(score >= THRESHOLD) if score is not None else None
            item["responses"][model] = {
                "text": text,
                **(response if isinstance(response, dict) else {}),
                "embedding_cosine_score": score,
                "embedding_cosine_verdict": verdict,
            }

            with open(INPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    print("Done. embedding_cosine_score / verdict added to raw_responses.json")


if __name__ == "__main__":
    run()
