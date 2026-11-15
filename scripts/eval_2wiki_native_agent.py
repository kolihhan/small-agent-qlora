"""Source-disjoint 2Wiki validation Qwen9B agent read/search baseline.

No GAIA labels, no model training, no teacher forcing. Source answers are
sealed until all autonomous episodes in a shard complete; the environment
contains original task-context pages and distractors, searchable locally.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
from itertools import islice
import json
from pathlib import Path
import re
from time import monotonic
from urllib.request import urlopen

from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.benchmark.scoring import gaia_score
from gaia_small_agent.model.ollama import OllamaModel
from gaia_small_agent.tools import ReadTool
from gaia_small_agent.tools.base import Tool, ToolResult
from gaia_small_agent.training.trajectories import _materialize_files
from scripts.audit_2wiki_search_read_paths import context_support

MODEL = "qwen3.5:9b-q4_K_M"
DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
REPO = "framolfese/2WikiMultihopQA"
REV = "fe713bfbd1afbca1a65246741a75890405d56a3a"
PROTOCOL = "2wiki-validation-qwen9b-search-read-baseline-v1"
N = 8
SHARDS = 4


def rank(s: str) -> str:
    return sha256(f"{PROTOCOL}:{s}".encode()).hexdigest()


def page_path(title: str) -> str:
    return "pages/" + sha256(title.encode()).hexdigest()[:12] + ".txt"


def freeze(rows: list[dict]) -> dict:
    cases = []
    used = set()
    for row in sorted(rows, key=lambda x: rank(str(x.get("id") or x.get("_id")))):
        context, support = context_support(row)
        identity = str(row.get("id") or row.get("_id") or "")
        if not identity or len(support) != 4 or len(context) < 6:
            continue
        if any(title in used for title in support):
            continue
        reference = str(row.get("answer") or "").strip()
        if not 1 <= len(reference) <= 80:
            continue
        # Conditional four-support, answer-source-visible cohort: not all 2Wiki.
        # This is checked with publisher labels in PRESELECTION, not agent prompt.
        if not any(reference.casefold() in " ".join(
            span for span in excerpts).casefold() for excerpts in support.values()):
            continue
        files = {page_path(t): text for t, text in context.items()}
        support_paths = {
            page_path(title): spans for title, spans in support.items()
        }
        used.update(support)
        cases.append({
            "id": identity,
            "question": row["question"],
            "answer": reference,
            "files": files,
            "index_titles": {title: page_path(title) for title in context},
            "support": support_paths,
        })
        if len(cases) == N:
            break
    if len(cases) != N:
        raise ValueError(f"insufficient 4-support VALIDATION tasks {len(cases)}")
    return {
        "protocol": PROTOCOL, "model_digest": DIGEST, "source_revision": REV,
        "split": "validation", "conditional_cohort": "4 supports, 6+ docs, reference literal in annotated source",
        "cases_sha": sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
        "cases": cases,
    }


class LocalSearch(Tool):
    name = "search"
    description = "Search local article titles/snippets by a named entity or topic. Returns article file paths for the read tool. Try another related name if no results."
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string", "minLength": 2}},
        "required": ["query"], "additionalProperties": False,
    }
    def __init__(self, title_paths: dict[str, str], files: dict[str, str]):
        self.paths, self.files = title_paths, files

    def run(self, arguments: dict, workspace: Path) -> ToolResult:
        query = str(arguments.get("query") or "").casefold().strip()
        tokens = set(re.findall(r"[a-z0-9]{3,}", query)) - {
            "what", "which", "that", "whose", "when", "where", "year", "name"
        }
        scored = []
        for title, path in self.paths.items():
            name = title.casefold()
            overlap = sum(token in name for token in tokens)
            phrase = int(bool(query and (query in name or name in query)))
            score = 4 * phrase + overlap
            if score:
                scored.append((score, title, path))
        scored.sort(key=lambda x: (-x[0], x[1].casefold()))
        results = [
            f"{title} | {path} | {self.files[path][:110]}"
            for _, title, path in scored[:5]
        ]
        return ToolResult(True, "\n".join(results) if results else "No matching local article found.")


def replay(case: dict, work: Path) -> dict:
    _materialize_files({"files": case["files"]}, work)
    # User only gets question, not a gold evidence index or source-answer.
    prompt = (
        "Answer the following question using the local article search and read tools. "
        "Search for article names and read returned paths before answering. "
        "A local source may be a distractor; decide whether more evidence is needed. "
        "Return only the answer, not reasoning.\n\nQuestion: " + case["question"]
    )
    model = OllamaModel(model=MODEL, timeout_s=400, max_new_tokens=512)
    start = monotonic()
    result = AgentRuntime(
        model, [LocalSearch(case["index_titles"], case["files"]), ReadTool()],
        max_steps=12,
    ).run(prompt, work)
    # Consume only genuine delivered READ observations, not retrieved path alone.
    pending = {}
    delivered = defaultdict(list)
    for event in result.trace:
        if event.kind == "tool_call" and event.data.get("name") == "read":
            path = event.data.get("arguments", {}).get("source")
            if isinstance(path, str):
                pending[event.data["id"]] = path
        if event.kind == "tool_result" and event.data.get("ok"):
            path = pending.get(event.data["id"])
            if path:
                obs = event.data.get("content", "")
                if obs.startswith("ToolMeta: "):
                    obs = obs.partition("\n")[2]
                delivered[path].append(obs)
    support = case["support"]
    full_support = all(
        any(all(span in msg for span in spans) for msg in delivered[path])
        for path, spans in support.items()
    )
    actions = [
        {"name": e.data["name"], "arguments": e.data["arguments"]}
        for e in result.trace if e.kind == "tool_call"
    ]
    return {
        "id": case["id"],
        "answer": result.answer,
        "completed": result.completed,
        "stop_reason": result.stop_reason,
        "tools": result.metrics.tool_calls,
        "tool_errors": result.metrics.tool_errors,
        "steps": result.metrics.steps,
        "full_support": full_support,
        "support_documents": len(support),
        "seconds": round(monotonic() - start, 1),
        "actions": actions,
    }


def eval_shard(cohort: dict, shard: int, work: Path) -> dict:
    if shard not in range(SHARDS):
        raise ValueError("bad shard")
    with urlopen("http://127.0.0.1:11434/api/tags", timeout=10) as response:
        manifests = json.load(response)["models"]
    if not any(x["name"] == MODEL and x["digest"] == DIGEST for x in manifests):
        raise ValueError("Qwen model identity changed")
    if cohort["source_revision"] != REV or cohort["protocol"] != PROTOCOL:
        raise ValueError("source/protocol changed")
    if sha256(json.dumps(cohort["cases"], sort_keys=True).encode()).hexdigest() != cohort["cases_sha"]:
        raise ValueError("cohort changed")
    cases = cohort["cases"][shard * 2:shard * 2 + 2]
    results = []
    for case in cases:
        got = replay(case, work / rank(case["id"])[:10])
        # Evaluate only after model makes unconditioned final decision.
        got["reference"] = case["answer"]
        got["exact"] = bool(got["completed"] and gaia_score(got["answer"], case["answer"]))
        results.append(got)
        print("2WIKI_QWEN_NATIVE", got["id"], got["exact"], got["full_support"],
              got["stop_reason"], got["tools"], flush=True)
    return {"protocol": PROTOCOL, "digest": DIGEST, "shard": shard,
            "cohort_sha": cohort["cases_sha"], "cases": results}


def aggregate(parts: list[dict]) -> dict:
    if (len(parts) != SHARDS or sorted(p["shard"] for p in parts) != list(range(SHARDS))
        or any(p["protocol"] != PROTOCOL or p["digest"] != DIGEST
               or len(p["cases"]) != 2 for p in parts)
        or len({p["cohort_sha"] for p in parts}) != 1):
        raise ValueError("incomplete Qwen native shards")
    rows = [r for part in parts for r in part["cases"]]
    if len({r["id"] for r in rows}) != N:
        raise ValueError("duplicate source identity")
    return {
        "protocol": PROTOCOL, "tasks": N, "digest": DIGEST,
        "strict_exact": sum(r["exact"] for r in rows),
        "completed": sum(r["completed"] for r in rows),
        "full_support": sum(r["full_support"] for r in rows),
        "wrong_with_full_support": sum(r["full_support"] and not r["exact"] for r in rows),
        "premature_final": sum(r["completed"] and not r["full_support"] for r in rows),
        "tool_calls": sum(r["tools"] for r in rows),
        "tool_errors": sum(r["tool_errors"] for r in rows),
        "model_training": False, "not_GAIA": True,
        "cases": rows,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--cohort", type=Path)
    p.add_argument("--shard", type=int)
    p.add_argument("--work", type=Path, default=Path("2wiki-work"))
    p.add_argument("--inputs", type=Path, nargs="+")
    p.add_argument("--split", type=str)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.inputs:
        data = aggregate([json.loads(x.read_text()) for x in a.inputs])
        print("2WIKI_QWEN_NATIVE_FINAL", json.dumps(
            {k: v for k, v in data.items() if k != "cases"}, sort_keys=True), flush=True)
    elif a.cohort:
        data = eval_shard(json.loads(a.cohort.read_text()), a.shard, a.work)
    else:
        if a.split != "validation":
            raise ValueError("freeze must use independent validation split")
        from datasets import load_dataset
        rows = list(islice(load_dataset(
            REPO, split="validation", revision=REV, streaming=True
        ), 1500))
        data = freeze(rows)
        print("2WIKI_QWEN_SOURCE_FROZEN", data["cases_sha"], flush=True)
    a.output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
