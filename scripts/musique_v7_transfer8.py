"""Matched Base/v7 MuSiQue TRAIN eight-task multi-hop reading probe (no training).

Uses the exact historical llama.cpp Base GGUF/v7 LoRA and the native
GAIA four-tool runtime. Score is local quasi-exact answer matching,
NOT the official MuSiQue leaderboard and NOT GAIA performance.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.benchmark.scoring import gaia_score
from gaia_small_agent.model.openai_compat import OpenAICompatModel
from gaia_small_agent.tools import default_tools
from gaia_small_agent.training.trajectories import _materialize_files

SHARDS = 4
COUNT = 8


def _load_tasks(path: Path) -> tuple[list[dict], str, str]:
    raw = path.read_bytes()
    tasks = [json.loads(x) for x in raw.decode("utf-8").splitlines() if x.strip()]
    if (len(tasks) != COUNT or len({r["id"] for r in tasks}) != COUNT
        or len({r["source_id_hash"] for r in tasks}) != COUNT
        or Counter(r["hop_count"] for r in tasks) != {3: 4, 4: 4}
        or any(r["source"] != "musique-train"
               or r["required_tools"] != ["read"]
               or len(r["files"]) != 21
               or "passages/index.txt" not in r["files"]
               or r["supporting_passages"] < r["hop_count"] for r in tasks)):
        raise ValueError("invalid frozen eight-task MuSiQue source")
    cohort = sha256("|".join(r["source_id_hash"] for r in tasks).encode()).hexdigest()
    return tasks, sha256(raw).hexdigest(), cohort


def run_shard(source: Path, shard: int) -> dict:
    if shard not in range(SHARDS):
        raise ValueError("invalid shard")
    tasks, source_sha, cohort_sha = _load_tasks(source)
    records = []
    with TemporaryDirectory() as tmp:
        for index, task in enumerate(tasks[shard::SHARDS]):
            folder = Path(tmp) / str(index)
            folder.mkdir()
            _materialize_files(task, folder)
            try:
                runtime = AgentRuntime(
                    OpenAICompatModel("qwen3.5-9b", max_new_tokens=512),
                    default_tools(), max_steps=12,
                )
                result = runtime.run(task["question"], folder)
                reads = {
                    str(ev.data.get("arguments", {}).get("source") or "")
                    for ev in result.trace
                    if ev.kind == "tool_call" and ev.data.get("name") == "read"
                }
                record = {
                    "id_hash": task["source_id_hash"],
                    "hop_count": task["hop_count"],
                    "correct": bool(result.completed and gaia_score(
                        result.answer, task["expected_answer"])),
                    "completed": bool(result.completed),
                    "stop": result.stop_reason,
                    "tool_calls": result.metrics.tool_calls,
                    "tool_errors": result.metrics.tool_errors,
                    "unique_passage_reads": len({
                        src for src in reads if src.startswith("passages/p")
                        and src.endswith(".txt")
                    }),
                    "index_read": "passages/index.txt" in reads,
                }
            except Exception as exc:
                record = {
                    "id_hash": task["source_id_hash"], "hop_count": task["hop_count"],
                    "correct": False, "completed": False,
                    "stop": "exception:" + type(exc).__name__,
                    "tool_calls": 0, "tool_errors": 0,
                    "unique_passage_reads": 0, "index_read": False,
                }
            records.append(record)
            print("MUSIQUE_MULTIHOP_TASK", shard, task["hop_count"],
                  record["correct"], record["stop"], flush=True)
    if len(records) != 2:
        raise ValueError("two source tasks per shard required")
    return {"shard": shard, "source_sha": source_sha,
            "cohort_sha": cohort_sha, "rows": records}


def aggregate(parts: list[dict]) -> dict:
    if len(parts) != 2 * SHARDS:
        raise ValueError("four Base and four v7 shards required")
    arms = {
        arm: sorted((r for r in parts if r.get("arm") == arm), key=lambda r: r["shard"])
        for arm in ("base", "v7")
    }
    if any([r["shard"] for r in arms[arm]] != list(range(SHARDS)) for arm in arms):
        raise ValueError("missing or duplicate model shard")
    if (len({r["source_sha"] for r in parts}) != 1
        or len({r["cohort_sha"] for r in parts}) != 1):
        raise ValueError("source or cohort mismatch")
    runs = {arm: [row for p in arms[arm] for row in p["rows"]] for arm in arms}
    if any(len(rows) != COUNT or len({r["id_hash"] for r in rows}) != COUNT
           for rows in runs.values()):
        raise ValueError("eight unique measured tasks needed")
    baseline, candidate = runs["base"], runs["v7"]
    if ([r["id_hash"] for r in baseline] != [r["id_hash"] for r in candidate]
        or [r["hop_count"] for r in baseline] != [r["hop_count"] for r in candidate]):
        raise ValueError("paired task identity mismatch")
    stats = {}
    for arm, rows in runs.items():
        stats[arm] = {
            "correct": sum(r["correct"] for r in rows),
            "correct_3hop": sum(r["correct"] for r in rows if r["hop_count"] == 3),
            "correct_4hop": sum(r["correct"] for r in rows if r["hop_count"] == 4),
            "completed": sum(r["completed"] for r in rows),
            "tool_calls": sum(r["tool_calls"] for r in rows),
            "tool_errors": sum(r["tool_errors"] for r in rows),
            "unique_passage_reads": sum(r["unique_passage_reads"] for r in rows),
            "reads_at_least_hops": sum(
                r["unique_passage_reads"] >= r["hop_count"] for r in rows
            ),
            "stop_reasons": dict(Counter(r["stop"] for r in rows)),
        }
    return {
        "n": COUNT, "arms": stats,
        "rescues": sum(not a["correct"] and b["correct"]
                       for a, b in zip(baseline, candidate)),
        "regressions": sum(a["correct"] and not b["correct"]
                           for a, b in zip(baseline, candidate)),
        "cohort_sha": parts[0]["cohort_sha"],
        "source_sha": parts[0]["source_sha"],
        "claim_boundary": (
            "MuSiQue TRAIN 4x3-hop + 4x4-hop, independent pinned source, "
            "20 passages per task, local conservative quasi-exact answers. "
            "Not official MuSiQue/GAIA; not new training or generalization proof."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path)
    p.add_argument("--shard", type=int)
    p.add_argument("--arm", choices=("base", "v7"))
    p.add_argument("--inputs", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("refuse overwrite")
    if a.inputs:
        result = aggregate([json.loads(p.read_text()) for p in a.inputs])
        print("MUSIQUE_PAIRED_FINAL", json.dumps(result, sort_keys=True), flush=True)
    else:
        if a.source is None or a.shard is None or a.arm is None:
            p.error("--source, --shard and --arm are required")
        result = run_shard(a.source, a.shard)
        result["arm"] = a.arm
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
