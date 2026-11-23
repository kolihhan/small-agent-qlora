"""Frozen disjoint MuSiQue DEV24B paired test of source-anchored evidence state.

No train examples, GAIA material, teacher trajectories or answer access in prompts.
Task IDs are ranks 13-24 per hop after prior DEV24A; the source family is not novel. Labels are scorer-only.
Never tune a tool using these results and repeat the same selected identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.benchmark.scoring import gaia_score
from gaia_small_agent.model.openai_compat import OpenAICompatModel
from gaia_small_agent.tools import default_tools
from gaia_small_agent.training.trajectories import _materialize_files

SOURCE_SHA256 = "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b"
SOURCE_REVISION = "667064b514d22deecae00a3dccbd92bccf88869e"
SEED = "independent-workspace-find-mu-dev24-20261123-frozen-v1"  # same ordering, second disjoint rank slice
PROTOCOL = "STEP100-SOURCE-ANCHORED-STATE-MUSIQUE-DEV24B-V1"
N = 24
SHARD_COUNT = 4
SHARD_N = 6


def select_rows(rows) -> list[dict]:
    """Select ranks 13-24 per hop, excluding previously frozen workspace-find first 12.

    Selection uses source IDs and structural validity, NEVER model outputs or correctness.
    """
    pool = {3: [], 4: []}
    seen = set()
    for row in rows:
        ident = str(row.get("id") or "")
        m = re.match(r"^([34])hop[0-9]+__", ident)
        if not m or ident in seen:
            continue
        seen.add(ident)
        hop = int(m[1])
        passages = row.get("paragraphs")
        decomp = row.get("question_decomposition")
        if (row.get("answerable", True) is not True
            or not str(row.get("question") or "").strip()
            or not str(row.get("answer") or "").strip()
            or not isinstance(passages, list) or len(passages) != 20
            or not isinstance(decomp, list) or len(decomp) != hop):
            continue
        indices = [p.get("idx") for p in passages if isinstance(p, dict)]
        if (len(indices) != 20 or len(set(indices)) != 20 or
            any(not isinstance(i, int) or isinstance(i, bool) or not 0 <= i <= 99 for i in indices)
            or any(not isinstance(p.get("paragraph_text"), str) or
                   not 20 <= len(p["paragraph_text"]) <= 9000 for p in passages)):
            continue
        rank = hashlib.sha256(f"{SEED}:{ident}".encode()).hexdigest()
        pool[hop].append((rank, ident, row))
    selected = []
    for hop in (3, 4):
        if len(pool[hop]) < 24:
            raise ValueError(f"Not enough valid DEV {hop}-hop rows; got {len(pool[hop])}")
        selected.extend([entry[2] for entry in sorted(pool[hop])[12:24]])
    return sorted(selected, key=lambda r: (hashlib.sha256(
        f"{SEED}:shuffle:{r['id']}".encode()).hexdigest(), str(r["id"])))


def load_frozen_selection(source: Path) -> tuple[list[dict], str]:
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Pinned MuSiQue DEV source physical SHA mismatch")
    with source.open(encoding="utf-8") as handle:
        selected = select_rows(json.loads(line) for line in handle if line.strip())
    if len(selected) != N or len({r["id"] for r in selected}) != N:
        raise ValueError("New DEV selection is incomplete")
    manifest = hashlib.sha256("\n".join(str(r["id"]) for r in selected).encode()).hexdigest()
    return selected, manifest


def workspace_task(row: dict) -> dict:
    files = {"passages/index.txt": "\n".join(
        f"passages/p{part['idx']}.txt | {part.get('title','')}"
        for part in sorted(row["paragraphs"], key=lambda x: x["idx"])
    )}
    for part in row["paragraphs"]:
        files[f"passages/p{part['idx']}.txt"] = part["paragraph_text"]
    if len(files) != 21:
        raise ValueError("Source workspace must include exactly 20 text passages plus index")
    return {
        "question": ("Answer the question using ONLY the workspace passage files. "
                     "Use the available tools to locate and read the evidence; "
                     "do not use public web search to find benchmark answers. "
                     "Question: " + str(row["question"])),
        "files": files,
        "expected_answer": str(row["answer"]),
    }


def run(source: Path, shard: int, arm: str, base_url: str, work_root: Path) -> dict:
    if shard not in range(SHARD_COUNT) or arm not in ("control", "state"):
        raise ValueError("Invalid frozen shard/arm")
    rows, manifest = load_frozen_selection(source)
    chosen = rows[shard * SHARD_N:(shard + 1) * SHARD_N]
    assert len(chosen) == SHARD_N
    model = OpenAICompatModel(model="qwen3.5-9b", base_url=base_url, max_new_tokens=512)
    cases = []
    for row in chosen:
        task = workspace_task(row)
        ident = str(row["id"])
        workspace = work_root / hashlib.sha256(ident.encode()).hexdigest()[:18]
        _materialize_files(task, workspace)
        agent = AgentRuntime(
            model, default_tools(),
            max_steps=12, evidence_state=(arm == "state"),
        )
        start = time.monotonic()
        result = agent.run(task["question"], workspace)
        elapsed = round(time.monotonic() - start, 3)
        cases.append({
            "id_sha256": hashlib.sha256((SEED + ":" + ident).encode()).hexdigest(),
            "hop_count": int(ident[0]), "correct": bool(result.completed and
                gaia_score(result.answer, task["expected_answer"])),
            "completed": bool(result.completed), "stop_reason": result.stop_reason,
            "tool_calls": result.metrics.tool_calls,
            "steps": result.metrics.steps,
            "tool_errors": result.metrics.tool_errors,
            "evidence_state_messages": sum("Evidence index:" in str((e.data or {}).get("content",""))
                for e in result.trace if e.kind=="tool_result"),
            "duplicate_calls": result.metrics.duplicate_calls_blocked,
            "elapsed_s": elapsed,
        })
    return {
        "protocol": PROTOCOL, "arm": arm, "shard": shard,
        "n": SHARD_N, "manifest_sha256": manifest,
        "source_sha256": SOURCE_SHA256,
        "adapter": "verified_qwen3.5_9b_v8_step100",
        "max_steps": 12, "max_new_tokens": 512,
        "only_variable": "opt_in_source_anchored_evidence_state",
        "answer_label_used_in_decisions": False,
        "no_task_or_answers_exported": True, "same_four_tools": True, "cases": cases,
    }


def score(directory: Path) -> dict:
    pairs, seen, manifest = [], set(), None
    for shard in range(SHARD_COUNT):
        arms = {}
        for arm in ("control", "state"):
            item = json.loads((directory/f"shard-{shard}-{arm}.json").read_text())
            if (item["protocol"] != PROTOCOL or item["n"] != SHARD_N or item["shard"] != shard
                or item["arm"] != arm or item["source_sha256"] != SOURCE_SHA256
                or item["only_variable"] != "opt_in_source_anchored_evidence_state"
                or item.get("same_four_tools") is not True
                or item["answer_label_used_in_decisions"] is not False
                or item["no_task_or_answers_exported"] is not True
                or item["max_steps"] != 12 or item["max_new_tokens"] != 512
                or item["adapter"] != "verified_qwen3.5_9b_v8_step100"):
                raise ValueError("Frozen pair protocol drift")
            if manifest is None:
                manifest = item["manifest_sha256"]
            if item["manifest_sha256"] != manifest:
                raise ValueError("Selection manifest drift")
            cases = {r["id_sha256"]:r for r in item["cases"]}
            if len(cases) != SHARD_N or any(
                r["hop_count"] not in (3, 4) or r["steps"] not in range(1, 13)
                or type(r["correct"]) is not bool or type(r["completed"]) is not bool
                or r["tool_calls"] < 0 or r["elapsed_s"] < 0
                for r in cases.values()
            ):
                raise ValueError("Duplicate or invalid cases")
            arms[arm] = cases
        if arms["control"].keys() != arms["state"].keys() or seen.intersection(arms["control"]):
            raise ValueError("Non-paired or repeated tasks")
        seen.update(arms["control"])
        pairs.extend((arms["control"][k], arms["state"][k]) for k in sorted(arms["control"]))
    if len(seen) != N:
        raise ValueError("Incomplete paired study")
    base = sum(a["correct"] for a,b in pairs)
    new = sum(b["correct"] for a,b in pairs)
    rescues = sum(not a["correct"] and b["correct"] for a,b in pairs)
    regressions = sum(a["correct"] and not b["correct"] for a,b in pairs)
    b_calls = sum(a["tool_calls"] for a,b in pairs)
    n_calls = sum(b["tool_calls"] for a,b in pairs)
    by_hop = {str(h): {
        "n": sum(a["hop_count"] == h for a,b in pairs),
        "base_correct":sum(a["correct"] for a,b in pairs if a["hop_count"] == h),
        "state_correct":sum(b["correct"] for a,b in pairs if a["hop_count"] == h),
        "rescues":sum(not a["correct"] and b["correct"] for a,b in pairs if a["hop_count"] == h),
        "regressions":sum(a["correct"] and not b["correct"] for a,b in pairs if a["hop_count"] == h),
    } for h in (3,4)}
    if new-base != rescues-regressions:
        raise ValueError("Score accounting mismatch")
    gate = {
        "net_at_least_3":new-base>=3,
        "regressions_max_1":regressions<=1,
        "positive_3hop_and_4hop":all(x["state_correct"]>x["base_correct"] for x in by_hop.values()),
        "call_cost_max_125pct":n_calls*4<=b_calls*5,
    }
    return {
        "protocol":PROTOCOL, "n":N,
        "source":"MuSiQue answerable DEV24B disjoint from active DEV24A and v8 TRAIN, fixed before outcome",
        "manifest_sha256":manifest, "source_sha256":SOURCE_SHA256,
        "base_correct":base,"state_correct":new,"rescues":rescues,"regressions":regressions,
        "base_calls":b_calls,"state_calls":n_calls,
        "base_elapsed_s":round(sum(a["elapsed_s"] for a,b in pairs),2),
        "state_elapsed_s":round(sum(b["elapsed_s"] for a,b in pairs),2),
        "by_hop":by_hop,"preregistered_gate":gate,"gate_pass":all(gate.values()),
        "control_index_messages":sum(a["evidence_state_messages"] for a,b in pairs),
        "state_index_messages":sum(b["evidence_state_messages"] for a,b in pairs),
        "not_GAIA100":True,
        "no_training_or_default_tool_change":True,
        "limitation":"MuSiQue source family in v8 TRAIN; IDs disjoint from v8 TRAIN and earlier DEV24A, but not an unseen-source-family test and may have historic DEV exposure. Do not tune on these rows.",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source", type=Path)
    p.add_argument("--shard", type=int)
    p.add_argument("--arm", choices=["control","state"])
    p.add_argument("--base-url", default="http://127.0.0.1:8080")
    p.add_argument("--work-root", type=Path)
    p.add_argument("--evidence-dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError("No overwrite of frozen evidence")
    data=(score(a.evidence_dir) if a.evidence_dir is not None else
          run(a.source,a.shard,a.arm,a.base_url,a.work_root))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(data,indent=2,sort_keys=True)+"\n")
    if a.evidence_dir is not None:
        print("EVIDENCE_STATE_DEV24B_OFFICIAL_PAIRED",json.dumps(data,sort_keys=True))
    else:
        print("EVIDENCE_STATE_DEV24B_SHARD",a.shard,a.arm,sum(x["correct"] for x in data["cases"]))
if __name__=="__main__":
    main()
