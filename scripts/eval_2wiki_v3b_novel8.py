"""Source-disjoint 2Wiki validation slice: physical Qwen3.5-9B Base vs v3B GGUF.

No GAIA labels, no source answers in model prompts. Same code, model and
local search/read tool surface; only LoRA adapter changes. Exploratory DEV.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import islice
import json
from pathlib import Path

import scripts.eval_2wiki_linked_read as linked
from scripts.eval_2wiki_native_agent import REPO, REV, MODEL, DIGEST

PROTOCOL = "qwen35-9b-v3b-vs-base-2wiki-novel-validation8-v1"
BASE_GGUF_SHA = "8c4cc24f54473f3d4e080d4ec7fd2a40d0bbec0f94f3c807cd62694905357d54"
ADAPTER_SHA = "e7caec609a905d21e056b107519d78916d3ef79e7bdc70db916e40d801d2187d"
N = 8
ARMS = ("base", "v3b")


def freeze_new(rows: list[dict]) -> dict:
    # Reproduce prior frozen source8 exactly, THEN screen them out along with
    # the originally exposed 2Wiki8 and all sixteen sets' support paths.
    previous = linked.freeze(rows[:1500])
    prev_ids = {c["id"] for c in previous["cases"]}
    previous_paths = {
        path.split("/")[-1][:-4]
        for c in previous["cases"] for path in c["support"]
    }
    linked.EXPOSED_IDS = frozenset(set(linked.EXPOSED_IDS) | prev_ids)
    linked.EXPOSED_SUPPORT_PATHS = frozenset(set(linked.EXPOSED_SUPPORT_PATHS) | previous_paths)
    selected = linked.freeze(rows)
    if {c["id"] for c in selected["cases"]} & prev_ids:
        raise ValueError("Leakage: a previous source8 task was selected")
    if any(path.split("/")[-1][:-4] in previous_paths
           for c in selected["cases"] for path in c["support"]):
        raise ValueError("Leakage: prior source support reused")
    return {
        "protocol": PROTOCOL, "source_revision": REV, "model": MODEL,
        "model_digest": DIGEST, "split": "validation",
        "n": N, "source8_sha": previous["cases_sha"],
        "new8_sha": selected["cases_sha"],
        "prior_recorded_task_ids_excluded": len(linked.EXPOSED_IDS),
        "cases": selected["cases"],
        "disclaimer": "Only task/support-disjoint from the previously recorded 16 VAL cases; not independent dataset or GAIA",
    }


def load_checked(path: Path) -> dict:
    p = json.loads(path.read_text(encoding="utf-8"))
    if p.get("protocol") != PROTOCOL or p.get("source_revision") != REV or p.get("n") != N:
        raise ValueError("Unrecognized frozen source")
    if p.get("model_digest") != DIGEST or p.get("source8_sha") != "fded3c65e2ccc829ae0efdd8af448896320590b63bb69e0e9ff1d52eadc4fbb7":
        raise ValueError("Original Qwen/source cohort drift")
    if p.get("new8_sha") != sha256(json.dumps(p["cases"], sort_keys=True).encode()).hexdigest():
        raise ValueError("Source cohort SHA mismatch")
    if len(p["cases"]) != N or len({c["id"] for c in p["cases"]}) != N:
        raise ValueError("Missing or duplicated test tasks")
    for c in p["cases"]:
        if len(c["files"]) < 6 or len(c["support"]) != 4 or not set(c["support"]) <= set(c["files"]):
            raise ValueError("Incomplete first-party source")
    return p


def evaluate_one(p: dict, index: int, arm: str, out: Path, base_url: str) -> dict:
    if index not in range(N) or arm not in ARMS:
        raise ValueError("Bad cohort task or arm")
    from gaia_small_agent.model.openai_compat import OpenAICompatModel

    def original_qwen_only(**_kw):
        return OpenAICompatModel(
            model="qwen3.5-9b", base_url=base_url, max_new_tokens=512,
        )
    linked.OllamaModel = original_qwen_only
    case = p["cases"][index]
    # Native arms only: original same local search/read, not the old linked treatment.
    result = linked.eval_arm(case, "native", out.parent / ("workspace-" + arm))
    report = {
        "protocol": PROTOCOL, "index": index, "task_id": case["id"], "arm": arm,
        "source8_sha": p["new8_sha"],
        "model": MODEL, "gguf_sha256": BASE_GGUF_SHA,
        "v3b_adapter_sha256": ADAPTER_SHA if arm == "v3b" else None,
        "result": result,
    }
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def aggregate(files: list[Path]) -> dict:
    if len(files) != 2 * N:
        raise ValueError("Expected precisely 16 physical arm outputs")
    all_rows = [json.loads(x.read_text()) for x in files]
    if any(r.get("protocol") != PROTOCOL or r.get("gguf_sha256") != BASE_GGUF_SHA for r in all_rows):
        raise ValueError("Model/protocol drift")
    if len({r["source8_sha"] for r in all_rows}) != 1:
        raise ValueError("Non-identical source cohort")
    by = {}
    for r in all_rows:
        key = (r["index"], r["arm"])
        if key in by: raise ValueError("Duplicated arm result")
        by[key] = r
    if set(by) != {(i, a) for i in range(N) for a in ARMS}:
        raise ValueError("Pairing mismatch")
    paired = [
        {"id": by[i, "base"]["task_id"],
         "base": by[i, "base"]["result"],
         "v3b": by[i, "v3b"]["result"]}
        for i in range(N)
    ]
    if any(by[i, "base"]["task_id"] != by[i, "v3b"]["task_id"] for i in range(N)):
        raise ValueError("Task identity changed between arms")
    base = sum(r["base"]["exact"] for r in paired)
    v3b = sum(r["v3b"]["exact"] for r in paired)
    rescue = sum(not r["base"]["exact"] and r["v3b"]["exact"] for r in paired)
    regressions = sum(r["base"]["exact"] and not r["v3b"]["exact"] for r in paired)
    return {
        "status": "complete", "protocol": PROTOCOL,
        "population": "2Wiki source task/support-disjoint VALIDATION8 (seen dataset)",
        "scorable": N, "base_exact": base, "v3b_exact": v3b,
        "rescue": rescue, "regression": regressions, "delta": v3b-base,
        "base_completed": sum(r["base"]["completed"] for r in paired),
        "v3b_completed": sum(r["v3b"]["completed"] for r in paired),
        "base_tool_calls": sum(r["base"]["tools"] for r in paired),
        "v3b_tool_calls": sum(r["v3b"]["tools"] for r in paired),
        "no_new_gaia_score": True, "promotable": False,
        "rows": paired,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", action="store_true")
    p.add_argument("--cohort", type=Path)
    p.add_argument("--index", type=int)
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--inputs", type=Path, nargs="+")
    p.add_argument("--base-url", default="http://127.0.0.1:8080")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if args.freeze:
        from datasets import load_dataset
        data = list(islice(load_dataset(REPO, split="validation", revision=REV, streaming=True), 2000))
        payload = freeze_new(data)
        print("NEW2WIKI_FROZEN", payload["new8_sha"], flush=True)
    elif args.inputs:
        payload = aggregate(args.inputs)
        print("V3B_NOVEL2WIKI_AGG", json.dumps({k:v for k,v in payload.items() if k!="rows"}), flush=True)
    else:
        if args.cohort is None or args.index is None or args.arm is None:
            p.error("--cohort --index --arm required")
        payload = evaluate_one(load_checked(args.cohort),args.index,args.arm,args.output,args.base_url)
        print("V3B_NOVEL2WIKI_CASE",args.index,args.arm,
              "exact",payload["result"]["exact"],flush=True)
        return
    args.output.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")


if __name__ == "__main__":
    main()
