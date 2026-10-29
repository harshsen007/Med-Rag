"""
eval.py — Ragas evaluation over three search modes.

Run:
  uv run python eval.py

What it does:
  1. Loads 17 evaluation questions from eval_questions.json
  2. For each mode (dense, sparse, hybrid):
       - Retrieves chunks and generates answers
       - Collects (question, answer, contexts, ground_truth) for Ragas
       - Scores with faithfulness, context_precision, context_recall
  3. Prints a comparison table

Requires:
  - A populated Qdrant collection (run ingest first)
  - NVIDIA_API_KEY in .env
  - eval_questions.json with a list of {"question": ..., "ground_truth": ...} objects
"""

import json
import os
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.config import settings
from src.retrieve import retrieve
from src.generate import answer

QUESTIONS_FILE = Path("eval_questions.json")


def load_questions() -> list[dict]:
    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(
            f"{QUESTIONS_FILE} not found.\n"
            "Create it with a list of {\"question\": ..., \"ground_truth\": ...} objects."
        )
    return json.loads(QUESTIONS_FILE.read_text())


def run_eval_for_mode(questions: list[dict], mode: str) -> dict:
    """
    Run retrieval + generation for every question under a given search mode.
    Returns a dict with Ragas metric scores.
    """
    print(f"\n── Mode: {mode} ────────────────────────────────")

    # Temporarily override SEARCH_MODE for this run
    settings.SEARCH_MODE = mode

    rows = []
    for i, item in enumerate(questions, start=1):
        q  = item["question"]
        gt = item["ground_truth"]
        print(f"  [{i}/{len(questions)}] {q[:60]}...")

        chunks = retrieve(q)
        ans    = answer(q, chunks=chunks)

        rows.append({
            "question":   q,
            "answer":     ans,
            "contexts":   [c["text"] for c in chunks],
            "ground_truth": gt,
        })

    dataset = Dataset.from_list(rows)

    # Use the NVIDIA endpoint for Ragas' internal LLM calls too
    ragas_llm = ChatOpenAI(
        model=settings.LLM_MODEL,
        openai_api_key=settings.NVIDIA_API_KEY,
        openai_api_base=settings.LLM_BASE_URL,
        temperature=0,
    )
    ragas_embeddings = OpenAIEmbeddings(
        model="nvidia/nv-embedqa-e5-v5",
        openai_api_key=settings.NVIDIA_API_KEY,
        openai_api_base=settings.LLM_BASE_URL,
    )

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    scores = {
        "faithfulness":       result["faithfulness"],
        "context_precision":  result["context_precision"],
        "context_recall":     result["context_recall"],
    }

    print(f"  faithfulness={scores['faithfulness']:.3f}  "
          f"precision={scores['context_precision']:.3f}  "
          f"recall={scores['context_recall']:.3f}")

    return scores


def main():
    questions = load_questions()
    print(f"Loaded {len(questions)} evaluation questions.")

    all_scores = {}
    for mode in ["dense", "sparse", "hybrid"]:
        all_scores[mode] = run_eval_for_mode(questions, mode)

    # Print comparison table
    print("\n\n── Results ─────────────────────────────────────────────────")
    print(f"{'Mode':<10} {'Faithfulness':>14} {'Precision':>12} {'Recall':>10}")
    print("-" * 52)
    for mode, scores in all_scores.items():
        print(
            f"{mode:<10} "
            f"{scores['faithfulness']:>14.3f} "
            f"{scores['context_precision']:>12.3f} "
            f"{scores['context_recall']:>10.3f}"
        )


if __name__ == "__main__":
    main()
