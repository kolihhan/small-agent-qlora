"""Base/v7 real-document numerical tool-use transfer on independent FinQA DEV.

20 distinct financial reports, multi-step annotated computations, labels checked
for internal numeric consistency. Matching questions and 4 native GAIA tools.
Questions, reasoning programs and gold answers never enter model prompts beyond
user question/report. Custom numeric scorer, NOT official FinQA leaderboard.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.model.openai_compat import OpenAICompatModel
from gaia_small_agent.tools import default_tools

N_SHARDS = 5
COUNT = 20
QUESTION_CUE = re.compile(
    r"\b(percent|ratio|average|difference|change|increase|decrease|"
    r"compared|growth|how much|how many|what proportion)\b", re.I,
)
NUMERIC = re.compile(r"^\s*\$?\s*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*(%)?\s*$")


def number(text: str) -> tuple[float, bool] | None:
    match = NUMERIC.fullmatch(str(text))
    if not match:
        return None
    return float(match.group(1).replace(",", "")), bool(match.group(2))


def close(a: float, b: float) -> bool:
    return math.isfinite(a) and math.isfinite(b) and math.isclose(
        a, b, rel_tol=0.001, abs_tol=0.02
    )


def score_numeric(answer: str, gold: str) -> bool:
    parsed, expected = number(answer), number(gold)
    if parsed is None or expected is None:
        return False
    value, percentage = parsed
    target, target_pct = expected
    if close(value, target):
        return True
    # 0.935 and 93.5% are equivalent only when the gold is a percentage.
    return bool(target_pct and not percentage and close(100 * value, target))


def report_text(item: dict) -> str:
    pre = item.get("pre_text") or []
    post = item.get("post_text") or []
    table = item.get("table") or []
    if not all(isinstance(s, str) for s in [*pre, *post]):
        raise ValueError("invalid financial text")
    if not isinstance(table, list) or not all(
        isinstance(row, list) and all(isinstance(cell, str) for cell in row)
        for row in table
    ):
        raise ValueError("invalid financial table")
    return (
        "FINANCIAL REPORT — CONTEXT BEFORE TABLE\n"
        + "\n".join(pre)
        + "\n\nFINANCIAL TABLE\n"
        + "\n".join(" | ".join(row) for row in table)
        + "\n\nFINANCIAL REPORT — CONTEXT AFTER TABLE\n"
        + "\n".join(post)
    )


def select_cases(source: Path) -> tuple[list[dict], str]:
    raw = source.read_bytes()
    source_hash = sha256(raw).hexdigest()
    rows = json.loads(raw)
    if not isinstance(rows, list) or len(rows) != 883:
        raise ValueError("FinQA DEV source changed (expected 883)")
    candidates = []
    for item in rows:
        qa = item.get("qa") or {}
        question = str(qa.get("question") or "")
        gold = number(str(qa.get("answer") or ""))
        exe = qa.get("exe_ans")
        steps = qa.get("steps") or []
        if (len(question) < 25 or not QUESTION_CUE.search(question)
            or not isinstance(exe, (float, int)) or isinstance(exe, bool)
            or gold is None or not isinstance(steps, list) or len(steps) < 2):
            continue
        value, pct = gold
        # Guard against internally inconsistent original FinQA annotation.
        if not (close(value, float(exe)) or (pct and close(value / 100, float(exe)))):
            continue
        evidence = report_text(item)
        if not 50 <= len(evidence) <= 12000:
            continue
        candidates.append(item)
    ordered = sorted(candidates, key=lambda item: (
        sha256(f"finqa-realnum-v7-20261114:{item['id']}".encode()).hexdigest(),
        str(item["id"]),
    ))
    result = []
    used_docs = set()
    for item in ordered:
        doc = str(item.get("filename") or "")
        if not doc or doc in used_docs:
            continue
        result.append(item)
        used_docs.add(doc)
        if len(result) == COUNT:
            break
    if len(result) != COUNT or len({r["id"] for r in result}) != COUNT:
        raise ValueError("missing 20 report-independent multi-step FinQA tasks")
    return result, source_hash


def run_shard(source: Path, shard: int) -> dict:
    if shard not in range(N_SHARDS):
        raise ValueError("invalid shard")
    cases, source_sha = select_cases(source)
    chosen = cases[shard::N_SHARDS]
    if len(chosen) != COUNT // N_SHARDS:
        raise ValueError("incomplete shard")
    hashes = [sha256(row["id"].encode()).hexdigest() for row in cases]
    records = []
    with TemporaryDirectory() as root:
        for i, row in enumerate(chosen):
            folder = Path(root) / str(i)
            folder.mkdir()
            (folder / "report.txt").write_text(report_text(row), encoding="utf-8")
            question = (
                "Use the local file report.txt as the ONLY evidence. Read the file. "
                "Use the python tool if arithmetic is required. Return ONLY the "
                "numeric answer (include % when the question requests a percentage).\n"
                "Question: " + row["qa"]["question"]
            )
            runtime = AgentRuntime(
                OpenAICompatModel("qwen3.5-9b", max_new_tokens=512),
                default_tools(), max_steps=12,
            )
            try:
                r = runtime.run(question, folder)
                records.append({
                    "id_hash": sha256(row["id"].encode()).hexdigest(),
                    "correct": bool(r.completed and score_numeric(
                        r.answer, str(row["qa"]["answer"])
                    )),
                    "completed": bool(r.completed),
                    "stop": r.stop_reason,
                    "tool_calls": r.metrics.tool_calls,
                    "tool_errors": r.metrics.tool_errors,
                    "duplicates": r.metrics.duplicate_calls_blocked,
                    "annotated_steps": len(row["qa"]["steps"]),
                    "backend_error": None,
                })
            except Exception as exc:
                records.append({
                    "id_hash": sha256(row["id"].encode()).hexdigest(),
                    "correct": False, "completed": False,
                    "stop": "exception", "tool_calls": 0,
                    "tool_errors": 0, "duplicates": 0,
                    "annotated_steps": len(row["qa"]["steps"]),
                    "backend_error": type(exc).__name__,
                })
            print("FINQA_NUMERIC_TASK", shard, i, records[-1]["correct"],
                  records[-1]["stop"], flush=True)
    return {
        "shard": shard, "source_sha": source_sha,
        "cohort_hash": sha256("|".join(hashes).encode()).hexdigest(),
        "rows": records,
    }


def aggregate(parts: list[dict]) -> dict:
    if len(parts) != 2 * N_SHARDS:
        raise ValueError("five shards per model required")
    arms = {
        arm: sorted((x for x in parts if x.get("arm") == arm), key=lambda x: x["shard"])
        for arm in ("base", "v7")
    }
    if any([r["shard"] for r in arms[a]] != list(range(N_SHARDS)) for a in arms):
        raise ValueError("missing/duplicate shard")
    if len({r["source_sha"] for r in parts}) != 1 or len({r["cohort_hash"] for r in parts}) != 1:
        raise ValueError("source or cohort differs")
    records = {arm: [r for x in arms[arm] for r in x["rows"]] for arm in arms}
    if any(len(rows) != COUNT or len({x["id_hash"] for x in rows}) != COUNT
           for rows in records.values()):
        raise ValueError("20 unique cases needed")
    before, after = records["base"], records["v7"]
    if [x["id_hash"] for x in before] != [x["id_hash"] for x in after]:
        raise ValueError("matched task identity drift")
    stats = {}
    for arm, rows in records.items():
        stats[arm] = {
            "correct": sum(x["correct"] for x in rows),
            "completed": sum(x["completed"] for x in rows),
            "tool_calls": sum(x["tool_calls"] for x in rows),
            "tool_errors": sum(x["tool_errors"] for x in rows),
            "duplicates": sum(x["duplicates"] for x in rows),
            "stop_reasons": dict(Counter(x["stop"] for x in rows)),
            "backend_errors": dict(Counter(
                x["backend_error"] for x in rows if x["backend_error"]
            )),
        }
    return {
        "n": COUNT, "arms": stats,
        "rescues": sum(not x["correct"] and y["correct"] for x, y in zip(before, after)),
        "regressions": sum(x["correct"] and not y["correct"] for x, y in zip(before, after)),
        "source_sha": parts[0]["source_sha"],
        "cohort_hash": parts[0]["cohort_hash"],
        "claim_boundary": "FinQA DEV real financial multi-step tasks, custom numeric scorer with original annotation consistency gate; not FinQA leaderboard, not GAIA, no training",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path)
    p.add_argument("--shard", type=int)
    p.add_argument("--arm", choices=("base", "v7"))
    p.add_argument("--inputs", nargs="+", type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if a.inputs:
        data = aggregate([json.loads(x.read_text()) for x in a.inputs])
        print("FINQA_TRANSFER_FINAL", json.dumps(data, sort_keys=True), flush=True)
    else:
        if a.source is None or a.shard is None or not a.arm:
            p.error("--source, --shard, --arm required")
        data = run_shard(a.source, a.shard)
        data["arm"] = a.arm
    a.output.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
