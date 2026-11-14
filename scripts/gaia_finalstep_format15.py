"""Same-prefix, bounded final-step-only answer formatting on independent GAIA15.

Keep every pre-terminal action and observation identical; only when the
12-step runtime forces a no-tools final request, append a generic system
instruction. Uses the original 180-second timeout, not 600 seconds.
The selected GAIA15 was used in a prior terminal suffix diagnostic, so this
is external replication, not a pristine new holdout.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

from gaia_small_agent.benchmark.gaia100 import load_validation
from gaia_small_agent.benchmark.sharding import shard_selected_rows
from gaia_small_agent.model.openai_compat import OpenAICompatModel
from gaia_small_agent.agent.types import ModelRuntimeError
from scripts.gaia_format_reminder_ablation import FORMAT_REMINDER
from scripts.gaia_terminal_instruction import TerminalProbe, independent_gaia, replay


N_SHARDS = 5


def final_only_messages(messages: list[dict]) -> list[dict]:
    if not messages or messages[0].get("role") != "system":
        raise ValueError("unexpected original transcript")
    return [{**messages[0], "content": messages[0]["content"] + FORMAT_REMINDER},
            *messages[1:]]


class FinalStepProbe(TerminalProbe):
    def __init__(self, model: str, base_url: str):
        super().__init__(model, base_url)
        self.terminal_model = OpenAICompatModel(
            model=model, base_url=base_url, timeout_s=180, max_new_tokens=512,
        )

    def complete_final(self, messages, tools):
        begin = perf_counter()
        try:
            original = self.terminal_model.complete_final(messages, tools)
        except ModelRuntimeError as exc:
            original = exc
        start = perf_counter()
        try:
            changed = self.terminal_model.complete_final(
                final_only_messages(messages), tools,
            )
        except ModelRuntimeError as exc:
            changed = exc
        end = perf_counter()
        self.terminals.append((changed, start - begin, end - start))
        if isinstance(original, ModelRuntimeError):
            raise original
        return original


def execute(shard: int) -> dict:
    if shard not in range(N_SHARDS):
        raise ValueError("invalid final-step shard")
    rows = shard_selected_rows(
        independent_gaia(load_validation()),
        shard_index=shard, shard_count=N_SHARDS,
    )
    if len(rows) != 3:
        raise ValueError("expected three cases per shard")
    paired = replay(rows, FinalStepProbe("qwen3.5-9b", "http://127.0.0.1:8080"))
    for row in paired:
        row["format_final_only"] = row.pop("explicit_final")
        if row["legacy"]["tool_calls"] != row["format_final_only"]["tool_calls"]:
            raise ValueError("tool decisions drifted before the final step")
    return {"shard": shard, "cases": paired}


def aggregate(shards: list[dict]) -> dict:
    if len(shards) != N_SHARDS or sorted(s["shard"] for s in shards) != list(range(N_SHARDS)):
        raise ValueError("need five GAIA15 shards")
    rows = [r for s in sorted(shards, key=lambda s: s["shard"]) for r in s["cases"]]
    ids = [r["task_hash"] for r in rows]
    if len(rows) != 15 or len(set(ids)) != 15:
        raise ValueError("expected unique GAIA15 cases")
    for row in rows:
        a, b = row["legacy"], row["format_final_only"]
        for key in ("tool_calls", "tool_errors", "duplicate_calls_blocked", "steps"):
            if a[key] != b[key]:
                raise ValueError("pre-terminal tool traces are not identical")
    arms = {}
    for condition in ("legacy", "format_final_only"):
        vals = [r[condition] for r in rows]
        arms[condition] = {
            **{k: sum(x[k] for x in vals) for k in
               ("correct", "completed", "tool_calls", "tool_errors", "duplicate_calls_blocked")},
            "stops": dict(Counter(x["stop_reason"] for x in vals)),
            "mean_wall_s": round(sum(x["latency_ms"] for x in vals) / 15000.0, 2),
            "terminal_cases": sum("terminal_ms" in x for x in vals),
            "terminal_mean_s": round(
                sum(x.get("terminal_ms", 0) for x in vals)
                / max(1, sum("terminal_ms" in x for x in vals)) / 1000, 2,
            ),
        }
    return {
        "cases": 15,
        "source": "GAIA15 outside Diagnostic25/Evaluation100/Shadow25, reused prior terminal experiment",
        "paired_rescues": sum(not r["legacy"]["correct"] and
                              r["format_final_only"]["correct"] for r in rows),
        "paired_regressions": sum(r["legacy"]["correct"] and
                                  not r["format_final_only"]["correct"] for r in rows),
        "arms": arms,
        "boundary": "same prefix and tools; final-only prompt no model training; timeout 180 seconds",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard", type=int)
    p.add_argument("--inputs", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.inputs:
        out = aggregate([json.loads(x.read_text()) for x in args.inputs])
        print("FINALSTEP_GAIA15", json.dumps(out, sort_keys=True), flush=True)
    else:
        if args.shard is None:
            p.error("--shard required")
        out = execute(args.shard)
        print("FINALSTEP_SHARD", args.shard, flush=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
