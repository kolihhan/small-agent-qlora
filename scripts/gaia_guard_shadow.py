"""Matched GAIA Shadow25 CPU Ollama check: original vs opt-in runtime tool guard.

Never export gated questions, answers or tool observations. Only publish SHA256
task IDs and aggregate behavioral/scored booleans. The frozen GAIA shadow set
is disjoint from the historical Evaluation100 and Diagnostic25 partitions.
Same model, 12-step budget, and exact tool set for both arms.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from time import perf_counter

from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.benchmark.gaia100 import GAIA_REVISION, LOCAL_PROTOCOL_SEED, load_validation, select_gaia_partition
from gaia_small_agent.benchmark.runner import known_capability_gap
from gaia_small_agent.benchmark.scoring import gaia_score
from gaia_small_agent.model.ollama import OllamaModel
from gaia_small_agent.tools import default_tools


def shard_rows(selected: list[dict], *, index: int, count: int = 4) -> list[dict]:
    if count != 4 or index not in range(4) or len(selected) != 25:
        raise ValueError("expected frozen Shadow25 / 4 CPU shards")
    ids = [str(row["task_id"]) for row in selected]
    if len(set(ids)) != 25:
        raise ValueError("duplicated shadow questions")
    return [row for i, row in enumerate(selected) if i % count == index]


def prepare_question(task: dict, workspace: Path) -> str:
    """Mimic canonical GAIA runner's user-visible task/attachment format."""
    question = str(task["Question"])
    file_path = task.get("file_path")
    if file_path:
        source = Path(str(file_path))
        if source.is_file():
            workspace.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, workspace / source.name)
        question += f"\nAn attachment is available in the workspace as {source.name}."
    question += (
        "\n\nBenchmark answer format: Return only the final answer as a number, "
        "as few words as possible, or a comma-separated list. Do not add explanation."
    )
    return question


def public_row(task: dict, result, *, elapsed_s: float) -> dict:
    """Release metrics not gated GAIA question/answer/trace contents."""
    return {
        "task_hash": hashlib.sha256(str(task["task_id"]).encode()).hexdigest(),
        "level": int(task["Level"]),
        "correct": bool(result.completed and gaia_score(result.answer, str(task["Final answer"]))),
        "completed": bool(result.completed),
        "stop_reason": result.stop_reason,
        "steps": int(result.metrics.steps),
        "tool_calls": int(result.metrics.tool_calls),
        "tool_errors": int(result.metrics.tool_errors),
        "duplicate_calls_blocked": int(result.metrics.duplicate_calls_blocked),
        "capability_gap": known_capability_gap(task.get("file_path")),
        "elapsed_s": round(elapsed_s, 3),
    }


def aggregate(shards: list[dict]) -> dict:
    if len(shards) != 4 or sorted(x.get("shard") for x in shards) != list(range(4)):
        raise ValueError("all four GAIA shards required")
    if any(x.get("schema") != "gaia-tool-guard-shadow-shard-v1" for x in shards):
        raise ValueError("wrong protocol")
    rows = [r for shard in shards for r in shard["rows"]]
    grouped: dict[str, dict[str, dict]] = {}
    for row in rows:
        if row.get("arm") not in {"baseline", "guard"}:
            raise ValueError("unexpected treatment")
        g = grouped.setdefault(str(row["task_hash"]), {})
        if row["arm"] in g:
            raise ValueError("repeated arm for task")
        g[row["arm"]] = row
    if len(grouped) != 25 or any(set(g) != {"baseline", "guard"} for g in grouped.values()):
        raise ValueError("not 25 paired tasks")
    def stats(arm):
        rr = [g[arm] for g in grouped.values()]
        return {
            "correct": sum(x["correct"] for x in rr),
            "completed": sum(x["completed"] for x in rr),
            "tool_calls": sum(x["tool_calls"] for x in rr),
            "duplicates_blocked": sum(x["duplicate_calls_blocked"] for x in rr),
            "model_steps": sum(x["steps"] for x in rr),
            "tool_errors": sum(x["tool_errors"] for x in rr),
        }
    b,g = stats("baseline"),stats("guard")
    gains=sum(not x["baseline"]["correct"] and x["guard"]["correct"] for x in grouped.values())
    losses=sum(x["baseline"]["correct"] and not x["guard"]["correct"] for x in grouped.values())
    if g["correct"]-b["correct"]!=gains-losses:
        raise AssertionError("paired accuracy mismatch")
    return {
        "schema":"gaia-tool-guard-shadow-aggregate-v1",
        "task_count":25,"baseline":b,"guard":g,
        "wrong_to_correct":gains,"correct_to_wrong":losses,
        "net_correct":g["correct"]-b["correct"],
        "new_training_ran":False,
        "model":"qwen3.5:9b-q4_K_M",
        "evaluation":"GAIA Shadow25, local fixed runner, no hidden leaderboard",
        "caveat":"Single fixed 25-task shadow; do not tune on these labels; no v3B adapter evaluated.",
        "rows":sorted(rows,key=lambda x:(x["task_hash"],x["arm"])),
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard",type=int)
    p.add_argument("--model",default="qwen3.5:9b-q4_K_M")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--inputs",type=Path,nargs="+")
    p.add_argument("--workspace",type=Path,default=Path("runs/gaia-guard-shadow"))
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError("refuse overwriting frozen GAIA evidence")
    if a.inputs:
        data=aggregate([json.loads(x.read_text(encoding="utf-8")) for x in a.inputs])
    else:
        if a.shard is None:
            p.error("shard required")
        dataset=load_validation(token=os.getenv("HF_TOKEN"),revision=GAIA_REVISION)
        selected=select_gaia_partition(dataset,partition="shadow",seed=LOCAL_PROTOCOL_SEED)
        shard=shard_rows(selected,index=a.shard)
        all_rows=[]
        for arm,guard in (("baseline",False),("guard",True)):
            model=OllamaModel(model=a.model,max_new_tokens=512,enable_thinking=False,timeout_s=300)
            runtime=AgentRuntime(model,default_tools(),max_steps=12,tool_policy_guard=guard)
            for i,task in enumerate(shard,1):
                workspace=a.workspace / arm / hashlib.sha256(str(task["task_id"]).encode()).hexdigest()
                workspace.mkdir(parents=True,exist_ok=True)
                question=prepare_question(task,workspace)
                t=perf_counter()
                result=runtime.run(question,workspace)
                record=public_row(task,result,elapsed_s=perf_counter()-t)
                record["arm"]=arm
                all_rows.append(record)
                print(f"GUARD_SHADOW shard={a.shard} arm={arm} {i}/{len(shard)} correct={record['correct']} completed={record['completed']}",flush=True)
        data={"schema":"gaia-tool-guard-shadow-shard-v1","shard":a.shard,
              "count":len(shard),"model":a.model,"rows":all_rows}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(data,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print("SHADOW_RESULT",json.dumps({k:v for k,v in data.items() if k!="rows"},sort_keys=True))


if __name__=="__main__":
    main()
