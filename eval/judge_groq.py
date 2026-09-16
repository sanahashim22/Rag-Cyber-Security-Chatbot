"""
judge_groq.py
-------------
Stage 3: LLM-as-judge evaluation using the Groq API.

For every (query, model_response) pair, asks a Groq-hosted LLM whether the
response matches the ground truth IN MEANING (not exact wording) -> Yes/No.

SETUP:
    setx GROQ_API_KEY "your-key-here"     (PowerShell, then restart terminal)
    or just set it for the current session:
    $env:GROQ_API_KEY="your-key-here"

USAGE:
    python judge_groq.py

OUTPUT:
    Adds a "groq_verdict" (1/0) and "groq_raw" (the model's Yes/No + reason)
    field to each response inside raw_responses.json.
"""

import json
import os
import time

from groq import Groq

INPUT_FILE = os.path.join(os.path.dirname(__file__), "raw_responses.json")
JUDGE_MODEL = "llama-3.3-70b-versatile"  # solid free-tier Groq judge model as of writing

JUDGE_SYSTEM_PROMPT = """You are a strict but fair grader for a cybersecurity Q&A system.
You will be given a QUESTION, a GROUND TRUTH answer, and a MODEL RESPONSE.

Judge whether the MODEL RESPONSE is correct based on MEANING, not exact wording.
The model response can be phrased completely differently, be longer or shorter,
or include extra correct detail -- as long as it captures the key fact(s) in the
ground truth and does not contradict them, it counts as correct.

Reply in exactly this format, nothing else:
VERDICT: Yes
or
VERDICT: No
REASON: <one short sentence>
"""


def build_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY environment variable not set. See the setup instructions at the top of this file.")
    return Groq(api_key=api_key)


def judge_one(client: Groq, query: str, ground_truth: str, model_response: str) -> dict:
    user_prompt = f"QUESTION: {query}\n\nGROUND TRUTH: {ground_truth}\n\nMODEL RESPONSE: {model_response}"
    try:
        completion = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=100,
        )
        raw = completion.choices[0].message.content.strip()
        verdict = 1 if "verdict: yes" in raw.lower() else 0
        return {"groq_verdict": verdict, "groq_raw": raw}
    except Exception as e:
        print(f"    Groq API error: {e}")
        return {"groq_verdict": None, "groq_raw": f"ERROR: {e}"}


def run():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

    client = build_client()

    for qid, item in results.items():
        query, ground_truth = item["query"], item["ground_truth"]
        for model, response in item["responses"].items():
            if isinstance(response, dict) and "groq_verdict" in response:
                continue  # already judged

            print(f"[{qid}] judging {model}...")
            judged = judge_one(client, query, ground_truth, response if isinstance(response, str) else response["text"])

            # Convert the plain string response into a dict that carries all judge scores.
            text = response if isinstance(response, str) else response["text"]
            item["responses"][model] = {"text": text, **judged}

            with open(INPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            time.sleep(0.3)  # be polite to the free tier rate limit

    print("Done. groq_verdict added to raw_responses.json")


if __name__ == "__main__":
    run()
