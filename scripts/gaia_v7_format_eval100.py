"""Preregistered replication: existing v7 format reminder on frozen GAIA Evaluation100.

All 100 cases are *seen GAIA validation DEV*, NOT an independent holdout.
Both arms use the identical original Qwen3.5-9B + verified v7 adapter,
runtime and tool budget; only the original generic system-format reminder differs.
No benchmark answers exposed outside temporary runner working dirs.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.benchmark import runner
from gaia_small_agent.benchmark.gaia100 import GAIA_REVISION, LOCAL_PROTOCOL_SEED, load_validation, select_gaia_partition
from gaia_small_agent.benchmark.sharding import shard_selected_rows
from gaia_small_agent.model.openai_compat import OpenAICompatModel
from gaia_small_agent.tools import default_tools

REMINDER = (
    "\nFollow the user's requested output format exactly. If they request only "
    "a code, number, value, or token, output that value alone without a sentence, "
    "Markdown, or explanation. Do not skip necessary tools or invent evidence.\n"
)
ARMS = ("original", "format_reminder")
N_SHARDS = 20

class FormatModel(OpenAICompatModel):
    def __init__(self, *, reminder: bool):
        super().__init__("qwen3.5-9b", max_new_tokens=512)
        self.reminder = reminder

    def _prepare(self, messages):
        if not self.reminder:
            return messages
        if not messages or messages[0].get("role") != "system":
            raise ValueError("Transcript must start with system")
        return [{**messages[0], "content": messages[0]["content"] + REMINDER}, *messages[1:]]

    def complete(self, messages, tools):
        return super().complete(self._prepare(messages), tools)

    def complete_final(self, messages, tools):
        return super().complete_final(self._prepare(messages), tools)


def select_fixed(dataset):
    rows = select_gaia_partition(dataset, partition="evaluation", seed=LOCAL_PROTOCOL_SEED)
    if len(rows) != 100 or len({r["task_id"] for r in rows}) != 100:
        raise ValueError("Expected original frozen GAIA Evaluation100")
    levels = Counter(int(r["Level"]) for r in rows)
    if levels != {1: 32, 2: 52, 3: 16}:
        raise ValueError("Frozen GAIA level distribution mismatch")
    return rows


def evaluate(rows, *, reminder):
    old = runner.select_gaia_partition
    with TemporaryDirectory(prefix="gaia-v7-replication-") as tmp:
        work = Path(tmp)
        try:
            runner.select_gaia_partition = lambda *_args, **_kwargs: rows
            runner.run_gaia100(
                rows,
                lambda: AgentRuntime(FormatModel(reminder=reminder), default_tools(), max_steps=12),
                work, seed=LOCAL_PROTOCOL_SEED, partition="evaluation", dataset_revision=GAIA_REVISION
            )
        finally:
            runner.select_gaia_partition = old
        records = [json.loads(s) for s in (work / "results.jsonl").read_text().splitlines() if s.strip()]
    if len(records) != len(rows):
        raise RuntimeError("Missing paired GAIA tasks")
    return [{
        "task_hash": sha256(r["task_id"].encode()).hexdigest(),
        "level": int(r["level"]),
        "correct": bool(r["correct"]),
        "completed": bool(r["completed"]),
        "stop_reason": r["stop_reason"],
        "tool_calls": int(r["tool_calls"]),
        "tool_errors": int(r["tool_errors"]),
        "duplicates": int(r["duplicate_calls_blocked"]),
        "latency_ms": round(float(r["latency_ms"]), 2),
        "capability_gap": bool(r.get("capability_gap")),
    } for r in records]


def run_shard(shard, dataset):
    if shard not in range(N_SHARDS):
        raise ValueError("Invalid shard index")
    selected = select_fixed(dataset)
    rows = shard_selected_rows(selected, shard_index=shard, shard_count=N_SHARDS)
    if len(rows) != 5:
        raise ValueError("Must have exactly five GAIA cases per shard")
    arms = {a: evaluate(rows, reminder=(a == "format_reminder")) for a in ARMS}
    if [x["task_hash"] for x in arms["original"]] != [x["task_hash"] for x in arms["format_reminder"]]:
        raise ValueError("Task ordering drift")
    return {
        "shard": shard, "source_revision": GAIA_REVISION,
        "selection": "gaia-evaluation100-seen-dev",
        "full_hash": sha256("|".join(str(r["task_id"]) for r in selected).encode()).hexdigest(),
        "arms": arms,
    }


def aggregate(shards):
    if len(shards) != N_SHARDS or sorted(x["shard"] for x in shards) != list(range(N_SHARDS)):
        raise ValueError("Need all 20 unique paired shards; partial results not allowed")
    first = shards[0]
    for s in shards:
        if s["source_revision"] != GAIA_REVISION or s["full_hash"] != first["full_hash"] or s["selection"] != first["selection"]:
            raise ValueError("Dataset identity drift")
        if any(len(s["arms"][a]) != 5 for a in ARMS):
            raise ValueError("Incomplete arm")
    by_arm = {a: [r for s in sorted(shards, key=lambda s:s["shard"]) for r in s["arms"][a]] for a in ARMS}
    orig, changed = by_arm["original"], by_arm["format_reminder"]
    ids = [r["task_hash"] for r in orig]
    if len(ids) != 100 or len(set(ids)) != 100 or ids != [r["task_hash"] for r in changed]:
        raise ValueError("Incorrect paired sample identity")
    stats = {}
    for arm, records in by_arm.items():
        stats[arm] = {
            **{k:sum(r[k] for r in records) for k in ("correct","completed","tool_calls","tool_errors","duplicates","capability_gap")},
            "mean_latency_s":round(sum(r["latency_ms"] for r in records)/100_000,2),
            "stop_reasons":dict(Counter(r["stop_reason"] for r in records)),
            "by_level":{str(l):{"correct":sum(r["correct"] for r in records if r["level"]==l),
                                "n":sum(r["level"]==l for r in records)} for l in (1,2,3)}
        }
    return {
        "status":"complete",
        "dataset":"GAIA validation, frozen already-seen Evaluation100 (NOT hidden holdout)",
        "model":"Qwen3.5-9B with exact v7 adapter (same in both arms)",
        "n":100,"arms":stats,
        "rescues":sum(not x["correct"] and y["correct"] for x,y in zip(orig,changed)),
        "regressions":sum(x["correct"] and not y["correct"] for x,y in zip(orig,changed)),
        "completion_rescues":sum(not x["completed"] and y["completed"] for x,y in zip(orig,changed)),
        "completion_regressions":sum(x["completed"] and not y["completed"] for x,y in zip(orig,changed)),
        "source_revision":GAIA_REVISION,
        "full_ids_sha256":first["full_hash"],
        "not_weight_training":True,
        "not_pristine":True,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--shard",type=int)
    p.add_argument("--inputs",type=Path,nargs="+")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.inputs:
        report=aggregate([json.loads(path.read_text()) for path in a.inputs])
        print("GAIA100_FINAL",json.dumps(report,sort_keys=True),flush=True)
    else:
        if a.shard is None: p.error("Need shard or inputs")
        report=run_shard(a.shard,load_validation())
        print("GAIA100_SHARD",a.shard, "complete",flush=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__":
    main()
