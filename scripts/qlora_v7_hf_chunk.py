"""Deterministic whole-parent free ZeroGPU micro-batches for QLoRA v7.

22 non-overlapping batches cover all 44 observed Hotpot/2Wiki TRAIN tasks and
all 174 synthetic policy tasks exactly once. Fixed source distribution:
16 batches with one task of each real source; six with two Hotpot tasks.
Synthetic per batch: one parent from each of eight capabilities when available.
No GAIA labels, no augmented copies of the same task, no chopped trajectories.
"""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256

_REAL_HOT = "hotpotqa-distractor-train"
_REAL_WIKI = "2wikimultihopqa-train"
_SYNTH = "synthetic-small-agent-policy-v3/train"
_CAPS = (
    "argument_grounding", "duplicate_avoidance", "evidence_to_final",
    "failure_recovery", "multi_step", "no_tool_stop", "strategy_switch",
    "tool_selection",
)
# 16 mixed-source / 6 hotpot-only batches, fixed before looking at task outcomes
_HOTPOT_DOUBLE = frozenset((3, 7, 11, 15, 19, 21))
_SEED = "qlora-v7-free-zero-gpu-real-parent-coverage-v1"
_TOTAL_BATCHES = 22


def _rank(value: str, namespace: str) -> str:
    return sha256(f"{_SEED}:{namespace}:{value}".encode()).hexdigest()


def build_free_gpu_chunk(rows: list[dict], batch_index: int) -> list[dict]:
    if type(batch_index) is not int or not 0 <= batch_index < _TOTAL_BATCHES:
        raise ValueError("batch_index must be 0..21")
    if not rows:
        raise ValueError("source rows required")
    if len({r.get("task_id") for r in rows}) != len(rows):
        raise ValueError("duplicate parent task IDs in complete training source")
    valid_sources = {_REAL_HOT, _REAL_WIKI, _SYNTH}
    if {r.get("source") for r in rows} != valid_sources:
        raise ValueError("source coverage must include both independent real TRAIN families and synthetic policy TRAIN")
    groups = defaultdict(list)
    for r in rows:
        if not isinstance(r.get("task_id"), str) or not r["task_id"]:
            raise ValueError("empty source task ID")
        indices = r.get("training_action_indices")
        if not isinstance(indices, list) or not indices or len(indices) != len(set(indices)):
            raise ValueError("invalid native assistant training actions")
        key = r["source"] if r["source"] != _SYNTH else "synth:" + str(r.get("capability") or "")
        groups[key].append(r)
    for key in groups:
        groups[key].sort(key=lambda r: (_rank(r["task_id"], key), r["task_id"]))
    if len(groups[_REAL_HOT]) != 28 or len(groups[_REAL_WIKI]) != 16:
        raise ValueError("source parents differ from frozen v7 28/16 verified split")
    if {key for key in groups if key.startswith("synth:")} != {"synth:" + cap for cap in _CAPS}:
        raise ValueError("unbalanced synthetic capability families")
    if any(len(groups["synth:" + cap]) not in {21,22} for cap in _CAPS):
        raise ValueError("synthetic source counts differ from frozen verified v7")
    prior_hot = sum(2 if i in _HOTPOT_DOUBLE else 1 for i in range(batch_index))
    prior_wiki = sum(0 if i in _HOTPOT_DOUBLE else 1 for i in range(batch_index))
    hot_count = 2 if batch_index in _HOTPOT_DOUBLE else 1
    wiki_count = 0 if batch_index in _HOTPOT_DOUBLE else 1
    selected = (
        groups[_REAL_HOT][prior_hot:prior_hot + hot_count] +
        groups[_REAL_WIKI][prior_wiki:prior_wiki + wiki_count]
    )
    if len(selected) != hot_count + wiki_count:
        raise ValueError("source-select real parents exhausted prematurely")
    for cap in _CAPS:
        items = groups["synth:" + cap]
        if batch_index < len(items):
            selected.append(items[batch_index])
    if len({r["task_id"] for r in selected}) != len(selected):
        raise AssertionError("chunk contains repeated training parent")
    return sorted(selected,key=lambda r:(_rank(r["task_id"],"learning-order"),r["task_id"]))
