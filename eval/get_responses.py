"""
get_responses.py
-----------------
Stage 2: Loop over the 5 models and get a response for every test query,
using the SAME retrieval + generation logic as RAGStream.py (via rag_core.py).

USAGE:
    Test with 1 query / 1 model first (recommended, matches your original plan):
        python get_responses.py --test

    Full run (all models, all queries):
        python get_responses.py

OUTPUT:
    raw_responses.json  -> {query_id: {query, ground_truth, responses: {model: text}}}
    (saved incrementally after every single response, so if it crashes or you
    Ctrl+C halfway through, you don't lose progress -- rerun and it skips
    anything already answered)
"""

import json
import os
import sys
import time
import argparse

from rag_core import ConvoRAGHeadless, chunk_text_with_overlap
from queries import TEST_QUERIES

# Make sure we can import load_documents.py from the RAGStream/ folder.
# This script is meant to live in an `eval/` folder alongside a copy of
# your project, e.g.:
#   latest-rag-version-2.0-eval/
#     RAGStream/
#       load_documents.py
#       docs/NIST.CSWP.29.pdf
#     eval/
#       get_responses.py   <- this file
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "RAGStream"))
from load_documents import load_all_from_docs_folder, list_loaded_documents  # noqa: E402

MODELS = ["llama3.2", "qwen2.5:1.5b", "gemma2:2b", "phi4-mini", "gemma4:e2b"]
EMBEDDING_MODEL = "nomic-embed-text"
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "raw_responses.json")


def load_documents_for_eval():
    """Same document loading RAGStream.py does: docs/ folder first, else built-in summary."""
    docs_folder_text = load_all_from_docs_folder()
    doc_names = list_loaded_documents()

    if docs_folder_text.strip():
        print(f"Loaded {len(doc_names)} document(s) from docs/ folder: {', '.join(doc_names)}")
        document_text = docs_folder_text
    else:
        # Falls back to the same built-in summary RAGStream.py uses.
        from RAGStream import DEFAULT_CYBERSECURITY_INFO  # only imported if needed
        print("No docs found in docs/ folder -- using built-in NIST CSF 2.0 summary.")
        document_text = DEFAULT_CYBERSECURITY_INFO

    chunks = chunk_text_with_overlap(document_text, chunk_size=200, overlap_size=40)
    print(f"Chunked into {len(chunks)} chunks.")
    return chunks


def load_existing_results():
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def run(test_mode: bool):
    documents = load_documents_for_eval()
    queries = TEST_QUERIES[:1] if test_mode else TEST_QUERIES
    models = MODELS[:1] if test_mode else MODELS

    results = load_existing_results()

    for model in models:
        print(f"\n{'=' * 60}\nModel: {model}\n{'=' * 60}")
        rag_engine = ConvoRAGHeadless(documents, embedding_model=EMBEDDING_MODEL, llm_model=model)

        for item in queries:
            qid, query, ground_truth = item["id"], item["query"], item["ground_truth"]
            results.setdefault(qid, {"query": query, "ground_truth": ground_truth, "responses": {}})

            if model in results[qid]["responses"]:
                print(f"  [{qid}] already answered by {model}, skipping.")
                continue

            print(f"  [{qid}] asking {model}: {query[:70]}...")
            start = time.time()
            rag_engine.conversation_history = []  # each query is independent for eval
            response = rag_engine.rag(query)
            elapsed = time.time() - start

            results[qid]["responses"][model] = response
            save_results(results)  # incremental save
            print(f"    -> got response in {elapsed:.1f}s ({len(response)} chars)")

    print(f"\nDone. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run only 1 query on 1 model, to sanity-check the pipeline first.")
    args = parser.parse_args()
    run(test_mode=args.test)
