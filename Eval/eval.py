"""
Eval/eval.py - Ragas evaluation of the MedRag pipeline over search modes.

Run from the project root:
    uv run python Eval/eval.py                       # all three modes, all questions, API judge
    uv run python Eval/eval.py --judge agy           # use the Antigravity CLI (agy) as judge
    uv run python Eval/eval.py --modes hybrid        # one mode
    uv run python Eval/eval.py --limit 3             # first 3 questions (quick test)
    uv run python Eval/eval.py --debug --limit 1     # crash with a full traceback on the first judge error

Needs:
  - a populated Qdrant collection (run ingest first)
  - NVIDIA_API_KEY (RAG LLM) and OPENROUTER_API_KEY (Ragas judge, unless --judge agy) in .env
  - for --judge agy: the `agy` CLI installed and signed in (see Eval/agy_judge.py)
  - eval_questions.json: [{"question": "...", "ground_truth": "..."}, ...]
    (looked for in Eval/ first, then the project root)

Output (in Eval/results/):
  - details_<mode>.csv : per-question scores, answers and retrieved chunks
  - summary.csv        : one row per mode
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# make `src` importable when running from anywhere
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import RunConfig, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import context_precision, context_recall, faithfulness

from src.config import settings
from src.generate import answer
from src.retrieve import retrieve

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
METRIC_COLUMNS = ["faithfulness", "context_precision", "context_recall"]

# Ragas run settings per judge. API judges are rate limited, so keep workers low.
# agy is a local subprocess per call (slow startup, but parallelises well).
RUN_SETTINGS = {
    "api": {"max_workers": 6, "timeout": 900, "max_retries": 10},
    "agy": {"max_workers": 8, "timeout": 600, "max_retries": 3},
}
for _cli in ("opencode", "openclaude", "claude", "gemini"):
    RUN_SETTINGS[_cli] = RUN_SETTINGS["agy"]


# ---------------------------------------------------------------
# Loading
# ---------------------------------------------------------------
def load_questions(limit: int | None) -> list[dict]:
    for path in (EVAL_DIR / "eval_questions.json", PROJECT_ROOT / "eval_questions.json"):
        if path.exists():
            questions = json.loads(path.read_text(encoding="utf-8"))
            break
    else:
        raise FileNotFoundError(
            "eval_questions.json not found in Eval/ or the project root.\n"
            'Create it as a list of {"question": ..., "ground_truth": ...} objects.'
        )
    return questions[:limit] if limit else questions


def build_judge(kind: str) -> LangchainLLMWrapper:
    if kind != "api":
        sys.path.insert(0, str(EVAL_DIR))
        from agy_judge import CliJudgeLLM

        return LangchainLLMWrapper(CliJudgeLLM(cli=kind))

    use_openrouter = "openrouter" in settings.R_LLM_BASE_URL
    api_key = settings.OPENROUTER_API_KEY if use_openrouter else settings.NVIDIA_API_KEY
    if not api_key:
        raise RuntimeError(
            "Missing API key for the judge: "
            + ("OPENROUTER_API_KEY" if use_openrouter else "NVIDIA_API_KEY")
        )
    llm = ChatOpenAI(
        model=settings.R_LLM_MODEL,
        api_key=api_key,
        base_url=settings.R_LLM_BASE_URL,
        temperature=0,
    )
    return LangchainLLMWrapper(llm)


# ---------------------------------------------------------------
# One mode: retrieve -> generate -> score
# ---------------------------------------------------------------
def run_mode(questions: list[dict], mode: str, judge, judge_kind: str, debug: bool) -> dict:
    print(f"\n-- Mode: {mode} " + "-" * 40)
    settings.SEARCH_MODE = mode

    rows = []
    for i, item in enumerate(questions, start=1):
        q, gt = item["question"], item["ground_truth"]
        print(f"  [{i}/{len(questions)}] {q[:70]}")
        chunks = retrieve(q)
        try:
            ans = answer(q, chunks=chunks)
        except Exception as exc:  # keep going if one generation call fails
            print(f"      answer failed: {exc!r}")
            ans = ""
        rows.append(
            {
                "question": q,
                "answer": ans or "",
                "contexts": [c["text"] for c in chunks],
                "ground_truth": gt,
            }
        )

    dataset = Dataset.from_list(rows)
    print(f"  Scoring with Ragas ({judge_kind} judge) ...")
    cfg = RUN_SETTINGS[judge_kind]
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, context_precision, context_recall],
        llm=judge,
        run_config=RunConfig(
            max_workers=cfg["max_workers"],
            timeout=cfg["timeout"],
            max_retries=cfg["max_retries"],
        ),
        raise_exceptions=debug,
    )

    df = result.to_pandas()
    for col in METRIC_COLUMNS:           # make sure every column exists
        if col not in df.columns:
            df[col] = np.nan
    df["retrieved_chunks"] = [" ||| ".join(r["contexts"]) for r in rows]

    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / f"details_{mode}.csv", index=False)

    summary = {"mode": mode, "questions": len(rows)}
    for col in METRIC_COLUMNS:
        values = df[col].astype(float)
        summary[col] = float(values.mean()) if values.notna().any() else float("nan")
        summary[f"{col}_nan"] = int(values.isna().sum())
    print(
        f"  faithfulness={fmt(summary['faithfulness'])}  "
        f"precision={fmt(summary['context_precision'])}  "
        f"recall={fmt(summary['context_recall'])}  "
        f"(NaN: {summary['faithfulness_nan']}/{summary['context_precision_nan']}/"
        f"{summary['context_recall_nan']})"
    )
    return summary


def fmt(v: float, width: int = 0) -> str:
    text = "NaN" if (v is None or math.isnan(v)) else f"{v:.3f}"
    return f"{text:>{width}}" if width else text


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Ragas evaluation for MedRag")
    parser.add_argument("--modes", nargs="+", default=["dense", "sparse", "hybrid"],
                        choices=["dense", "sparse", "hybrid"])
    parser.add_argument("--judge", choices=["api", "agy", "opencode", "openclaude", "claude", "gemini"],
                        default="api",
                        help="api = OpenRouter/NVIDIA from settings; otherwise an agent CLI")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--debug", action="store_true",
                        help="raise the first judge error instead of recording NaN")
    args = parser.parse_args()

    questions = load_questions(args.limit)
    print(f"Loaded {len(questions)} evaluation questions.")
    if args.judge != "api":
        print(f"Judge: {args.judge} CLI")
    else:
        print(f"Judge: {settings.R_LLM_MODEL} via {settings.R_LLM_BASE_URL}")
    print(f"RAG LLM: {settings.LLM_MODEL} | top_k={settings.TOP_K} | temperature={settings.TEMPERATURE}")

    judge = build_judge(args.judge)
    original_mode = settings.SEARCH_MODE
    summaries = []
    try:
        for mode in args.modes:
            summaries.append(run_mode(questions, mode, judge, args.judge, args.debug))
    finally:
        settings.SEARCH_MODE = original_mode

    # ---- comparison table ----
    print("\n\n== Results " + "=" * 58)
    print(f"{'Mode':<8} {'Faithfulness':>13} {'Precision':>10} {'Recall':>8}   NaN (f/p/r)")
    print("-" * 66)
    for s in summaries:
        print(
            f"{s['mode']:<8} {fmt(s['faithfulness'], 13)} {fmt(s['context_precision'], 10)} "
            f"{fmt(s['context_recall'], 8)}   "
            f"{s['faithfulness_nan']}/{s['context_precision_nan']}/{s['context_recall_nan']}"
        )
    print("\nNaN = judge call failed (timeout / rate limit / bad JSON). "
          "Scores with many NaNs are not reliable; re-run or lower max_workers.")

    import pandas as pd

    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(summaries).to_csv(RESULTS_DIR / "summary.csv", index=False)
    print(f"Saved details and summary to {RESULTS_DIR}")


if __name__ == "__main__":
    main()