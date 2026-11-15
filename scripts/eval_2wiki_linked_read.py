"""PRO paired original-Qwen native 2Wiki validation source-link affordance.

Not train data. Compare UNCHANGED AgentRuntime + original local title search and
ReadTool to ReadTool with bounded *original-text-mentioned* local page pointers.
No supporting-fact labels or answers enter the treatment; paired selection
excludes all eight previously exposed validation identities and supporting docs.
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

from gaia_small_agent.agent.loop import AgentRuntime, MAX_TOOL_OBSERVATION_CHARS
from gaia_small_agent.benchmark.scoring import gaia_score
from gaia_small_agent.model.ollama import OllamaModel
from gaia_small_agent.tools import ReadTool
from gaia_small_agent.tools.base import ToolResult
from gaia_small_agent.training.trajectories import _materialize_files
from scripts.eval_2wiki_native_agent import LocalSearch, page_path, MODEL, DIGEST, REPO, REV
from scripts.audit_2wiki_search_read_paths import context_support

PROTOCOL = "2wiki-validation-qwen9b-visible-local-links-pair-v1"
N = 8
SHARDS = 4
ARMS = ("native", "linked")
# The previous original native Qwen first8 validation source identities:
EXPOSED_IDS = frozenset({
    "1822cbb4089111ebbd72ac1f6bf848b6", "2e419977088111ebbd6dac1f6bf848b6",
    "36fc43f808dc11ebbd9cac1f6bf848b6", "d956416e086711ebbd5eac1f6bf848b6",
    "5bec3cd408a711ebbd7fac1f6bf848b6", "c5601df108a411ebbd7dac1f6bf848b6",
    "ef7aac32088511ebbd6dac1f6bf848b6", "a91c363708e711ebbda6ac1f6bf848b6",
})
EXPOSED_SUPPORT_PATHS = frozenset([
    "00dc3fb25000", "13f9d4c0bda6", "1bb3836144f4", "1df0580c169a",
    "20bb20c7de01", "25149a7ded22", "26e744bf956f", "296bd082a284",
    "3cbbabfd5858", "3e460acefa6c", "40ba7f92a16a", "50c4069bb759",
    "57a46389ea3c", "590895f88ed4", "6024fad10db5", "61bacc3869f2",
    "74468bcadf09", "750bd7f28412", "a0f5da80c7bd", "b072d5e25ba1",
    "b761b57f29e1", "b953c51803fe", "bf3f09ef073a", "d27b66f348ef",
    "d30458571101", "d4cb27906a6d", "d78db1dabb77", "e3796d7f11b1",
    "ec1356cf0af7", "f5f04a7ed1ca", "f754e4bbde83", "fe13d2a87d57",
])


def rank(s: str) -> str:
    return sha256((PROTOCOL + ":" + s).encode()).hexdigest()


def freeze(rows: list[dict]) -> dict:
    cases = []
    used_support = set(EXPOSED_SUPPORT_PATHS)
    for row in sorted(rows, key=lambda x: rank(str(x.get("id") or x.get("_id")))):
        identity = str(row.get("id") or row.get("_id") or "")
        if not identity or identity in EXPOSED_IDS:
            continue
        context, support = context_support(row)
        if len(support) != 4 or len(context) < 6:
            continue
        support_ids = {page_path(t).split("/")[-1][:-4] for t in support}
        if support_ids & used_support:
            continue
        answer = str(row.get("answer") or "").strip()
        if not 1 <= len(answer) <= 80:
            continue
        # Same selection mechanism as frozen first source cohort. This is
        # dataset feasibility and never used in any model prompt/tool.
        if not any(answer.casefold() in " ".join(spans).casefold()
                   for spans in support.values()):
            continue
        used_support.update(support_ids)
        cases.append({
            "id": identity, "question": row["question"], "answer": answer,
            "files": {page_path(t): body for t, body in context.items()},
            "index_titles": {t: page_path(t) for t in context},
            "support": {page_path(t): spans for t, spans in support.items()},
        })
        if len(cases) == N:
            break
    if len(cases) != N:
        raise ValueError(f"too few genuinely support-disjoint original VALIDATION tasks {len(cases)}")
    assert not set(c["id"] for c in cases) & EXPOSED_IDS
    assert not set(p.split("/")[-1][:-4] for c in cases for p in c["support"]) & EXPOSED_SUPPORT_PATHS
    return {
        "protocol": PROTOCOL, "model_digest": DIGEST, "source_revision": REV,
        "split": "validation",
        "selection": "four original supporting documents, >=6 local docs, answer in annotated source; IDs and support titles disjoint from prior 8",
        "cases_sha": sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
        "cases": cases,
    }


class LinkedRead(ReadTool):
    """Provide local document pointers only if actual visible source text names them."""
    def __init__(self, index: dict[str, str]):
        super().__init__()
        self.index = index

    def run(self, arguments: dict, workspace: Path) -> ToolResult:
        response = super().run(arguments, workspace)
        source = str(arguments.get("source") or "")
        if not response.ok or source not in self.index.values():
            return response
        # Only text that would already be visible within native 2000-char tool
        # observations may disclose a pointer; no oracle reference and no
        # peek into hidden later offsets.
        bounded_source = response.content[:1200]
        candidates = []
        for title, path in self.index.items():
            if path == source or len(title) < 4:
                continue
            match = re.search(r"(?<!\w)" + re.escape(title) + r"(?!\w)",
                              bounded_source, flags=re.I)
            if match:
                candidates.append((match.start(), title, path))
        candidates.sort(key=lambda x: (x[0], x[1].casefold()))
        if not candidates:
            return response
        metadata = dict(response.metadata)
        metadata["related_local_source_paths"] = [
            f"{title} | {path}" for _, title, path in candidates[:4]
        ]
        return ToolResult(response.ok, response.content, response.error_code, metadata)


def delivered(trace, support: dict) -> dict:
    calls, read = {}, defaultdict(list)
    for event in trace:
        if event.kind == "tool_call" and event.data.get("name") == "read":
            source = event.data.get("arguments", {}).get("source")
            if isinstance(source, str):
                calls[event.data["id"]] = source
        elif event.kind == "tool_result" and event.data.get("ok"):
            path = calls.get(event.data["id"])
            if path in support:
                text = event.data.get("content", "")
                if text.startswith("ToolMeta: "):
                    text = text.partition("\n")[2]
                read[path].append(text)
    visible = {
        path: any(all(span.casefold() in obs.casefold() for span in spans)
                  for obs in read[path])
        for path, spans in support.items()
    }
    return {
        "visible_support_count": sum(visible.values()),
        "full_support": all(visible.values()),
        "support_documents": len(support),
        "read_support_paths": sorted(path for path, yes in visible.items() if yes),
    }


def eval_arm(case: dict, arm: str, work: Path) -> dict:
    if arm not in ARMS:
        raise ValueError(arm)
    _materialize_files({"files": case["files"]}, work)
    prompt = (
        "Answer the following question using the local article search and read tools. "
        "Search for article names and read returned paths before answering. "
        "A local source may be a distractor; decide whether more evidence is needed. "
        "Return only the answer, not reasoning.\n\nQuestion: " + case["question"]
    )
    model = OllamaModel(model=MODEL, timeout_s=400, max_new_tokens=512)
    read = ReadTool() if arm == "native" else LinkedRead(case["index_titles"])
    start = monotonic()
    answer = AgentRuntime(
        model, [LocalSearch(case["index_titles"], case["files"]), read],
        max_steps=12
    ).run(prompt, work)
    evidence = delivered(answer.trace, case["support"])
    return {
        "answer": answer.answer,
        "completed": answer.completed,
        "stop_reason": answer.stop_reason,
        "tools": answer.metrics.tool_calls,
        "tool_errors": answer.metrics.tool_errors,
        "steps": answer.metrics.steps,
        "seconds": round(monotonic() - start, 1),
        "exact": bool(answer.completed and gaia_score(answer.answer, case["answer"])),
        **evidence,
        "actions": [
            {"name": e.data["name"], "arguments": e.data["arguments"]}
            for e in answer.trace if e.kind == "tool_call"
        ],
    }


def one_shard(cohort: dict, shard: int, root: Path) -> dict:
    if shard not in range(SHARDS):
        raise ValueError("bad shard")
    if (cohort["source_revision"] != REV or cohort["protocol"] != PROTOCOL
        or cohort["model_digest"] != DIGEST
        or sha256(json.dumps(cohort["cases"], sort_keys=True).encode()).hexdigest()
        != cohort["cases_sha"]):
        raise ValueError("cohort changed")
    with urlopen("http://127.0.0.1:11434/api/tags", timeout=10) as response:
        models = json.load(response)["models"]
    if not any(x["name"] == MODEL and x["digest"] == DIGEST for x in models):
        raise ValueError("original Qwen physical digest drift")
    cases = cohort["cases"][shard * 2:shard * 2 + 2]
    results = []
    for k, case in enumerate(cases):
        order = ARMS if (shard + k) % 2 == 0 else ARMS[::-1]
        out = {a: eval_arm(case, a, root / rank(case["id"])[:10] / a) for a in order}
        results.append({"id": case["id"], "reference": case["answer"], **out})
        print("2WIKI_SOURCE_LINKS_PAIRED", case["id"],
              {a: (out[a]["exact"], out[a]["full_support"],out[a]["tools"]) for a in ARMS},
              flush=True)
    return {"protocol": PROTOCOL, "digest": DIGEST, "source_revision": REV,
            "cohort_sha": cohort["cases_sha"], "shard": shard, "cases": results}


def aggregate(shards: list[dict]) -> dict:
    if (len(shards) != SHARDS or sorted(s["shard"] for s in shards) != list(range(SHARDS))
        or len({s["cohort_sha"] for s in shards}) != 1
        or any(s["protocol"] != PROTOCOL or s["digest"] != DIGEST
               or s["source_revision"] != REV or len(s["cases"]) != 2 for s in shards)):
        raise ValueError("incomplete paired physical source test")
    rows = [r for s in sorted(shards, key=lambda s: s["shard"]) for r in s["cases"]]
    if len({r["id"] for r in rows}) != N:
        raise ValueError("duplicate task IDs")
    metrics = {a: {
        "exact": sum(r[a]["exact"] for r in rows),
        "full_support": sum(r[a]["full_support"] for r in rows),
        "finished_without_full_support": sum(r[a]["completed"] and not r[a]["full_support"] for r in rows),
        "completed": sum(r[a]["completed"] for r in rows),
        "tool_calls": sum(r[a]["tools"] for r in rows),
        "tool_errors": sum(r[a]["tool_errors"] for r in rows),
        "seconds": round(sum(r[a]["seconds"] for r in rows),1),
    } for a in ARMS}
    rescues = [r["id"] for r in rows if not r["native"]["exact"] and r["linked"]["exact"]]
    regressions = [r["id"] for r in rows if r["native"]["exact"] and not r["linked"]["exact"]]
    evidence_gain = metrics["linked"]["full_support"] - metrics["native"]["full_support"]
    screen = (
        len(rescues) - len(regressions) >= 2 and not regressions and evidence_gain >= 2
        and metrics["linked"]["finished_without_full_support"] <=
            metrics["native"]["finished_without_full_support"]
        and metrics["linked"]["completed"] >= metrics["native"]["completed"]
    )
    return {
        "protocol": PROTOCOL, "tasks": len(rows), "source_revision": REV,
        "digest": DIGEST, "arms": metrics, "rescues": rescues, "regressions": regressions,
        "net_full_evidence_delta": evidence_gain, "positive_screen_only": screen,
        "promotion_allowed": False, "model_training": False, "not_GAIA": True,
        "original_tool_surface": "native title search and read",
        "treatment_tool_surface": "same native tools, source-mentioned title-path metadata on read",
        "cases": rows
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", type=Path)
    p.add_argument("--shard", type=int)
    p.add_argument("--inputs", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--work", type=Path, default=Path("linked-val-work"))
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.inputs:
        result = aggregate([json.loads(x.read_text()) for x in a.inputs])
        print("2WIKI_LINKED_READ_FINAL", json.dumps(
            {k: v for k,v in result.items() if k != "cases"}, sort_keys=True), flush=True)
    elif a.cohort:
        if a.shard is None:
            p.error("requires shard for physically paired Qwen inference")
        result = one_shard(json.loads(a.cohort.read_text()), a.shard, a.work)
    else:
        from datasets import load_dataset
        rows = list(islice(load_dataset(
            REPO, split="validation", revision=REV, streaming=True
        ), 1500))
        result = freeze(rows)
        print("2WIKI_LINKED_SOURCE_FREEZE", result["cases_sha"], flush=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
