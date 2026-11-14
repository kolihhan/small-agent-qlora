"""Paired terminal-only diagnostic on the fixed 15-case GAIA cohort.

The legacy rollout supplies the same observations to both last requests.
The intervention adds one explicit instruction and executes no extra tools.
Only task hashes and numeric or categorical metrics leave the runner.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from gaia_small_agent.agent.loop import AgentRuntime, _looks_like_textual_tool_call
from gaia_small_agent.agent.types import AssistantTurn, ModelRuntimeError
from gaia_small_agent.benchmark import runner
from gaia_small_agent.benchmark.gaia100 import GAIA_REVISION, LOCAL_PROTOCOL_SEED, load_validation, select_gaia_partition
from gaia_small_agent.benchmark.scoring import gaia_score
from gaia_small_agent.benchmark.sharding import shard_selected_rows
from gaia_small_agent.model.openai_compat import OpenAICompatModel
from gaia_small_agent.tools import default_tools


def terminal_messages(messages: list[dict]) -> list[dict]:
    return [*messages, {
        "role": "user",
        "content": (
            "This is the final response. No more tools are available. "
            "Use the observations you have and return only the requested answer."
        ),
    }]


def final_outcome(turn: AssistantTurn | ModelRuntimeError) -> tuple[str, str]:
    if isinstance(turn, ModelRuntimeError):
        return "", turn.stop_reason
    if turn.tool_calls:
        return "", "max_steps"
    answer = (turn.content or "").strip()
    if not answer:
        return "", "empty_final"
    if _looks_like_textual_tool_call(answer):
        return "", "invalid_final"
    return answer, "final"


class TerminalProbe(OpenAICompatModel):
    def __init__(self, model: str, base_url: str):
        super().__init__(model=model, base_url=base_url, max_new_tokens=512)
        self.terminal_model = OpenAICompatModel(
            model=model, base_url=base_url, timeout_s=600, max_new_tokens=512,
        )
        self.terminals: list[tuple[AssistantTurn | ModelRuntimeError, float, float]] = []

    def complete_final(self, messages, tools):
        begin = perf_counter()
        try:
            original = self.terminal_model.complete_final(messages, tools)
        except ModelRuntimeError as exc:
            original = exc
        start = perf_counter()
        try:
            changed = self.terminal_model.complete_final(terminal_messages(messages), tools)
        except ModelRuntimeError as exc:
            changed = exc
        end = perf_counter()
        self.terminals.append((changed, start - begin, end - start))
        if isinstance(original, ModelRuntimeError):
            raise original
        return original


def independent_gaia(dataset: list[dict]) -> list[dict]:
    seen = {
        row["task_id"] for split in ("diagnostic", "evaluation", "shadow")
        for row in select_gaia_partition(dataset, split)
    }
    selected = sorted((row for row in dataset if row["task_id"] not in seen),
                      key=lambda row: row["task_id"])
    if len(selected) != 15 or len({row["task_id"] for row in selected}) != 15:
        raise ValueError("expected 15 unique GAIA cases outside all three historical partitions")
    return selected


def replay(rows: list[dict], model: TerminalProbe) -> list[dict]:
    original_selector = runner.select_gaia_partition
    with TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        try:
            runner.select_gaia_partition = lambda *args, **kwargs: rows
            runner.run_gaia100(rows, lambda: AgentRuntime(model, default_tools(), max_steps=12),
                               workspace, seed=LOCAL_PROTOCOL_SEED)
        finally:
            runner.select_gaia_partition = original_selector
        results = [json.loads(line) for line in (workspace / "results.jsonl").read_text().splitlines()]

    terminal_index = 0
    reference = {row["task_id"]: str(row["Final answer"]) for row in rows}
    paired = []
    for row in results:
        case_id = row["task_id"]
        baseline = {k: row[k] for k in (
            "correct", "completed", "stop_reason", "tool_calls", "tool_errors",
            "duplicate_calls_blocked", "steps", "latency_ms",
        )}
        baseline["premature_final_proxy"] = "premature_final_proxy" in row["failure_signals"]
        changed = baseline.copy()
        if row["steps"] == 12:
            turn, legacy_s, changed_s = model.terminals[terminal_index]
            terminal_index += 1
            # Remove the opposite arm's request from the shared rollout time.
            changed["latency_ms"] -= legacy_s * 1000
            baseline["latency_ms"] -= changed_s * 1000
            baseline["terminal_ms"] = round(legacy_s * 1000, 2)
            changed["terminal_ms"] = round(changed_s * 1000, 2)
            answer, reason = final_outcome(turn)
            changed.update(correct=bool(answer and gaia_score(answer, reference[case_id])),
                           completed=reason == "final", stop_reason=reason)
        paired.append({
            "task_hash": sha256(case_id.encode()).hexdigest(), "level": row["level"],
            "legacy": baseline, "explicit_final": changed,
        })
    if terminal_index != len(model.terminals) or len(results) != len(rows):
        raise ValueError("rollout and terminal request evidence did not match")
    return paired


def aggregate(shards: list[dict]) -> dict:
    if sorted(s["shard"] for s in shards) != list(range(5)):
        raise ValueError("five independent GAIA shards required")
    rows = {arm: [r for s in shards for r in s[arm]] for arm in ("base", "v7")}
    if any(len(rr) != 15 or len({r["task_hash"] for r in rr}) != 15 for rr in rows.values()):
        raise ValueError("expected 15 unique cases per model")
    if [r["task_hash"] for r in rows["base"]] != [r["task_hash"] for r in rows["v7"]]:
        raise ValueError("Base and v7 task order differs")
    arms = {}
    for arm, rr in rows.items():
        stats = {}
        for condition in ("legacy", "explicit_final"):
            p = [r[condition] for r in rr]
            stats[condition] = {
                **{k: sum(r[k] for r in p) for k in (
                    "correct", "completed", "tool_calls", "tool_errors",
                    "duplicate_calls_blocked", "premature_final_proxy",
                )},
                "terminal_cases": sum(r["steps"] == 12 for r in p),
                "terminal_mean_ms": round(sum(r.get("terminal_ms", 0.0) for r in p)
                                          / max(1, sum("terminal_ms" in r for r in p)), 2),
                "mean_latency_ms": round(sum(r["latency_ms"] for r in p) / len(p), 2),
                "stop_reasons": dict(Counter(r["stop_reason"] for r in p)),
            }
        stats["paired"] = {
            "rescues": sum(not r["legacy"]["correct"] and r["explicit_final"]["correct"] for r in rr),
            "regressions": sum(r["legacy"]["correct"] and not r["explicit_final"]["correct"] for r in rr),
        }
        arms[arm] = stats
    return {"n_per_model": 15, "gaia_revision": GAIA_REVISION, "arms": arms}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("base", "v7"))
    parser.add_argument("--shard", type=int)
    parser.add_argument("--inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.inputs:
        result = aggregate([json.loads(p.read_text()) for p in args.inputs])
    else:
        if args.shard is None or args.arm is None:
            parser.error("--shard and --arm are required for inference")
        rows = shard_selected_rows(independent_gaia(load_validation()),
                                   shard_index=args.shard, shard_count=5)
        result = {"shard": args.shard,
                  args.arm: replay(rows, TerminalProbe("qwen3.5-9b", "http://127.0.0.1:8080"))}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
