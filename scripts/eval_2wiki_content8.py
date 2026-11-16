"""Real Qwen3.5-9B agent source-search ablation on new 2Wiki validation8.

Same base model, prompt, limits, source files and tool schema; change only
search matching of query terms against genuine local article body text.
No fine-tuning, no adapters, and no benchmark answers in tool observations.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
from urllib.request import urlopen

import scripts.eval_2wiki_linked_read as old
from scripts.eval_2wiki_v3b_novel8 import load_checked
from scripts.eval_2wiki_native_agent import MODEL, DIGEST
from gaia_small_agent.tools.base import ToolResult

PROTOCOL = "qwen35-9b-body-index-search-2wiki-newval8-paired-v1"
ARMS = ("native", "body_index")
N = 8
STOP = frozenset(
    "what which that whose when where year name does did the and are for who "
    "whom with from into have had has was were been being this those them "
    "some many most first last between after before about whose where".split()
)

def words(txt: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]{3,}", txt.casefold()) if w not in STOP]


class BodyIndexSearch(old.LocalSearch):
    """Exact same native tool schema and response; broader local corpus matching."""
    def __init__(self, title_paths: dict[str, str], files: dict[str, str]):
        super().__init__(title_paths, files)
        self.doc_counts = {path: Counter(words(files[path])) for path in title_paths.values()}
        self.titles = {path: title for title, path in title_paths.items()}
        self.df = Counter()
        for frequencies in self.doc_counts.values():
            self.df.update(frequencies.keys())

    def run(self, arguments: dict, workspace: Path) -> ToolResult:
        q = str(arguments.get("query") or "").strip()
        terms = set(words(q))
        if not terms:
            return super().run(arguments, workspace)
        n = len(self.doc_counts)
        scored = []
        for path, counts in self.doc_counts.items():
            title = self.titles[path]
            title_low = title.casefold()
            title_match = 4 * int(q.casefold() in title_low or title_low in q.casefold())
            title_match += 2 * sum(t in words(title) for t in terms)
            weighted = sum(
                math.log1p(n / (1 + self.df[t])) * min(5, counts[t])
                for t in terms if counts[t]
            )
            score = title_match + 0.85 * weighted
            if score > 0:
                scored.append((score, title, path))
        scored.sort(key=lambda x: (-x[0], x[1].casefold()))
        hits = [f"{title} | {path} | {self.files[path][:110]}"
                for _, title, path in scored[:5]]
        return ToolResult(True, "\n".join(hits) if hits else "No matching local article found.")


def assert_model() -> None:
    with urlopen("http://127.0.0.1:11434/api/tags", timeout=10) as response:
        models = json.load(response)["models"]
    if not any(r.get("name") == MODEL and r.get("digest") == DIGEST for r in models):
        raise RuntimeError("Original Qwen3.5-9B model digest drift")


def one(cohort: dict, index: int, arm: str, output: Path) -> dict:
    if arm not in ARMS or index not in range(N):
        raise ValueError("Invalid task/arm")
    assert_model()
    if arm == "body_index":
        old.LocalSearch = BodyIndexSearch
    case = cohort["cases"][index]
    score = old.eval_arm(case, "native", output.parent / ("workspace-" + str(index) + "-" + arm))
    record = {
        "protocol": PROTOCOL, "cohort_sha": cohort["new8_sha"],
        "index": index, "task_id": case["id"], "arm": arm,
        "model": MODEL, "model_digest": DIGEST,
        "score": score, "training": False,
    }
    output.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf8")
    print("REAL_2WIKI_AGENT_SEARCH", index, arm, "EXACT", int(score["exact"]),
          "FULL_SUPPORT", int(score["full_support"]), flush=True)
    return record


def aggregate(inputs: list[Path]) -> dict:
    if len(inputs) != 16:
        raise ValueError("Expected 16 actual pair-arm artifacts")
    items = [json.loads(path.read_text()) for path in inputs]
    if len({x.get("cohort_sha") for x in items}) != 1:
        raise ValueError("Source drift")
    if any(x.get("protocol") != PROTOCOL or x.get("model_digest") != DIGEST for x in items):
        raise ValueError("Protocol or Qwen model drift")
    by = {}
    for x in items:
        key = (x["index"], x["arm"])
        if key in by: raise ValueError("Duplicate case/arm")
        by[key] = x
    if set(by) != {(i, a) for i in range(N) for a in ARMS}:
        raise ValueError("Incomplete frozen pairs")
    rows = []
    for i in range(N):
        base, body = by[i,"native"], by[i,"body_index"]
        if base["task_id"] != body["task_id"]:
            raise ValueError("Task ID mismatch")
        rows.append({"id": base["task_id"], "baseline": base["score"], "body_index": body["score"]})
    a = sum(x["baseline"]["exact"] for x in rows)
    b = sum(x["body_index"]["exact"] for x in rows)
    rescues = sum(not x["baseline"]["exact"] and x["body_index"]["exact"] for x in rows)
    regressions = sum(x["baseline"]["exact"] and not x["body_index"]["exact"] for x in rows)
    return {
        "status": "complete", "protocol": PROTOCOL, "benchmark": "2Wiki validation8, source-disjoint from recorded prior16",
        "model": MODEL, "model_digest": DIGEST,
        "n": N, "base_correct": a, "treatment_correct": b, "delta": b-a,
        "rescues": rescues, "regressions": regressions,
        "base_full_support": sum(x["baseline"]["full_support"] for x in rows),
        "treatment_full_support": sum(x["body_index"]["full_support"] for x in rows),
        "base_completed": sum(x["baseline"]["completed"] for x in rows),
        "treatment_completed": sum(x["body_index"]["completed"] for x in rows),
        "base_tool_calls": sum(x["baseline"]["tools"] for x in rows),
        "treatment_tool_calls": sum(x["body_index"]["tools"] for x in rows),
        "gaia_improvement": False, "qlora_adapter_improvement": False,
        "not_independent_source": True, "not_promoted": True,
        "rows": rows,
    }


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--cohort", type=Path)
    p.add_argument("--index", type=int)
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--inputs", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError("refuse overwrite")
    if args.inputs:
        result=aggregate(args.inputs)
        print("REAL_2WIKI_SEARCH_PAIRED", json.dumps({k:v for k,v in result.items() if k!="rows"}), flush=True)
        args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf8")
    else:
        if args.index is None or args.arm is None or args.cohort is None:
            p.error("case evaluation requires --cohort --index --arm")
        one(load_checked(args.cohort),args.index,args.arm,args.output)


if __name__=="__main__":
    main()
