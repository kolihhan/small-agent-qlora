"""Source-shift test: Qwen3.5-9B evidence-link metadata and output serialization.

The treatment supplies only document-title pointers mentioned in already visible
local read observations (no gold/support labels). Raw and gold-blind last-line
GAIA-exact scores are both predeclared. HotpotQA public validation is not GAIA
and not an independent hidden test; no adapter, no model swap or teacher.
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
from scripts.audit_2wiki_answer_format_confound import last_line_answer

MODEL = "qwen3.5:9b-q4_K_M"
DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
REPO = "hotpotqa/hotpot_qa"
REV = "1908d6afbbead072334abe2965f91bd2709910ab"
PROTOCOL = "qwen9b-hotpot-distractor-val8-linked-read-plus-locked-final-v1"
N, SHARDS = 8, 4
ARMS = ("native", "linked")
# Locked before running any Hotpot agent inference:
GATE = {"net_end_to_end": 2, "regressions": 0, "net_evidence": 2,
        "max_call_ratio": 1.30}


def rank(key: str) -> str:
    return sha256((PROTOCOL + ":" + key).encode()).hexdigest()


def page_path(title: str) -> str:
    return "pages/" + sha256(title.encode()).hexdigest()[:12] + ".txt"


def freeze(rows: list[dict]) -> dict:
    by_type: dict[str, list[dict]] = defaultdict(list)
    used_titles = set()
    selected = []
    for row in sorted(rows, key=lambda x: rank(str(x.get("id")))):
        kind = str(row.get("type") or "")
        if kind not in ("bridge", "comparison"):
            continue
        if sum(c["type"] == kind for c in selected) >= N // 2:
            continue
        identity = str(row.get("id") or "")
        context = row.get("context") or {}
        titles, contents = context.get("title", []), context.get("sentences", [])
        if (not identity or len(titles) != len(contents) or
            len(titles) < 8 or len(set(titles)) != len(titles)):
            continue
        facts = row.get("supporting_facts") or {}
        supports = list(zip(facts.get("title", []), facts.get("sent_id", [])))
        support_titles = {title for title, _ in supports}
        if len(support_titles) != 2 or not support_titles.issubset(set(titles)):
            continue
        if support_titles & used_titles:
            continue
        sentences_by_title = dict(zip(titles, contents))
        support_spans = defaultdict(list)
        valid = True
        for title, index in supports:
            parts = sentences_by_title[title]
            if not isinstance(index, int) or not 0 <= index < len(parts):
                valid = False
                break
            support_spans[page_path(title)].append(str(parts[index]))
        if not valid or any(not x.strip() for spans in support_spans.values() for x in spans):
            continue
        answer = str(row.get("answer") or "")
        if not answer.strip() or len(answer) > 80:
            continue
        used_titles.update(support_titles)
        case = {
            "id": identity, "type": kind, "question": str(row["question"]),
            "answer": answer,
            "files": {page_path(t): t + "\n" + "\n".join(map(str, s))
                      for t, s in zip(titles, contents)},
            "index_titles": {t: page_path(t) for t in titles},
            "support": dict(support_spans),
        }
        selected.append(case)
        if len(selected) == N:
            break
    if len(selected) != N or {c["type"] for c in selected} != {"bridge", "comparison"}:
        raise RuntimeError("Hotpot source freeze could not find balanced 8 tasks")
    if len({c["id"] for c in selected}) != N:
        raise RuntimeError("Duplicate Hotpot identity")
    # There is no published GAIA answer in the model prompt, tool index or page metadata.
    case_sha = sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()
    return {
        "status": "FROZEN_SOURCE_BEFORE_MODEL",
        "protocol": PROTOCOL, "source_repo": REPO, "source_revision": REV,
        "source_split": "distractor/validation",
        "model_digest": DIGEST, "cases_sha256": case_sha,
        "n": N, "groups": {"bridge": N // 2, "comparison": N // 2},
        "predeclared_gate": GATE,
        "selection": "first 800 source rows; deterministic ID hash, >=8 distractor docs, two verified support-title identities, unique support titles across tasks; no inference outcomes",
        "public_pretraining_overlap_unknown": True,
        "cases": selected,
    }


class LocalSearch(Tool):
    name = "search"
    description = "Search local article titles and snippets by a named entity or topic. Returns file paths for reading."
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string", "minLength": 2}},
        "required": ["query"], "additionalProperties": False,
    }

    def __init__(self, index: dict[str, str], files: dict[str, str]):
        self.index, self.files = index, files

    def run(self, arguments: dict, workspace: Path) -> ToolResult:
        query = str(arguments.get("query") or "").casefold().strip()
        tokens = set(re.findall(r"[a-z0-9]{3,}", query)) - {
            "what", "which", "that", "whose", "when", "where", "year", "name"
        }
        scored = []
        for title, path in self.index.items():
            name = title.casefold()
            overlap = sum(t in name for t in tokens)
            phrase = int(bool(query and (query in name or name in query)))
            score = 4 * phrase + overlap
            if score:
                scored.append((score, title, path))
        scored.sort(key=lambda x: (-x[0], x[1].casefold()))
        found = [f"{title} | {path} | {self.files[path][:110]}"
                 for _, title, path in scored[:5]]
        return ToolResult(True, "\n".join(found) if found else
                          "No matching local article found.")


class LinkedRead(ReadTool):
    """Append bounded links named by already visible source text, not oracle supports."""
    def __init__(self, index: dict[str, str]):
        super().__init__()
        self.index = index

    def run(self, arguments: dict, workspace: Path) -> ToolResult:
        response = super().run(arguments, workspace)
        source = str(arguments.get("source") or "")
        if not response.ok or source not in self.index.values():
            return response
        visible = response.content[:1200]
        links = []
        for title, path in self.index.items():
            if path == source or len(title) < 4:
                continue
            mention = re.search(
                r"(?<!\w)" + re.escape(title) + r"(?!\w)",
                visible, flags=re.IGNORECASE)
            if mention:
                links.append((mention.start(), title, path))
        links.sort(key=lambda x: (x[0], x[1].casefold()))
        if not links:
            return response
        metadata = dict(response.metadata)
        metadata["related_local_source_paths"] = [
            f"{title} | {path}" for _, title, path in links[:4]]
        return ToolResult(response.ok, response.content, response.error_code, metadata)


def delivered(trace, support: dict) -> int:
    calls, reads = {}, defaultdict(list)
    for event in trace:
        if event.kind == "tool_call" and event.data.get("name") == "read":
            p = event.data.get("arguments", {}).get("source")
            if isinstance(p, str):
                calls[event.data["id"]] = p
        elif event.kind == "tool_result" and event.data.get("ok"):
            p = calls.get(event.data["id"])
            if p in support:
                body = str(event.data.get("content") or "")
                if body.startswith("ToolMeta: "):
                    body = body.partition("\n")[2]
                reads[p].append(body)
    return sum(any(all(span.casefold() in body.casefold() for span in spans)
                   for body in reads[path]) for path, spans in support.items())


def evaluate(case: dict, arm: str, root: Path) -> dict:
    if arm not in ARMS:
        raise ValueError(arm)
    _materialize_files({"files": case["files"]}, root)
    tools = [LocalSearch(case["index_titles"], case["files"]),
             ReadTool() if arm == "native" else LinkedRead(case["index_titles"])]
    prompt = (
        "Answer using the local article search and read tools. "
        "Search for relevant article titles, read source paths before answering, "
        "and judge whether the evidence suffices. "
        "Some articles are distractors. Return only the answer.\n\nQuestion: " +
        case["question"]
    )
    started = monotonic()
    result = AgentRuntime(
        OllamaModel(model=MODEL, timeout_s=400, max_new_tokens=512),
        tools, max_steps=12
    ).run(prompt, root)
    raw = str(result.answer or "")
    cleaned = last_line_answer(raw)
    return {
        "raw_answer": raw, "cleaned_answer": cleaned,
        "completed": result.completed, "stop_reason": result.stop_reason,
        "raw_exact": bool(result.completed and gaia_score(raw, case["answer"])),
        "locked_lastline_exact": bool(
            result.completed and gaia_score(cleaned, case["answer"])),
        "support_read_count": delivered(result.trace, case["support"]),
        "all_two_supports_read": delivered(result.trace, case["support"]) == 2,
        "tool_calls": result.metrics.tool_calls,
        "tool_errors": result.metrics.tool_errors,
        "duplicate_calls": result.metrics.duplicate_calls_blocked,
        "seconds": round(monotonic() - started, 1),
    }


def run_shard(cohort: dict, shard: int, work: Path) -> dict:
    if shard not in range(SHARDS) or len(cohort.get("cases", [])) != N:
        raise ValueError("Invalid cohort or shard")
    if (cohort.get("status") != "FROZEN_SOURCE_BEFORE_MODEL" or
        cohort.get("protocol") != PROTOCOL or
        cohort.get("source_revision") != REV or
        cohort.get("model_digest") != DIGEST or
        cohort.get("predeclared_gate") != GATE or
        sha256(json.dumps(cohort["cases"], sort_keys=True).encode()).hexdigest()
        != cohort.get("cases_sha256")):
        raise ValueError("Source SHA / gate drift")
    with urlopen("http://127.0.0.1:11434/api/tags", timeout=15) as response:
        models = json.load(response)["models"]
    if not any(x["name"] == MODEL and x["digest"] == DIGEST for x in models):
        raise RuntimeError("Qwen model digest drift")
    subset = cohort["cases"][shard * 2: shard * 2 + 2]
    scored = []
    for i, case in enumerate(subset):
        order = ARMS if (i + shard) % 2 == 0 else ARMS[::-1]
        by_arm = {arm: evaluate(case, arm,
                    work / rank(case["id"])[:8] / arm) for arm in order}
        scored.append({"id": case["id"], "type": case["type"], "arms": by_arm})
        print("HOTPOT_PAIRED_CASE", case["id"],
              {a: (by_arm[a]["raw_exact"], by_arm[a]["locked_lastline_exact"],
                   by_arm[a]["support_read_count"]) for a in ARMS}, flush=True)
    return {
        "status": "complete", "protocol": PROTOCOL, "source_sha": cohort["cases_sha256"],
        "model_digest": DIGEST, "shard": shard, "n": len(scored), "cases": scored,
    }


def aggregate(parts: list[dict]) -> dict:
    if (len(parts) != SHARDS or sorted(x.get("shard") for x in parts) != list(range(SHARDS))
        or any(x.get("status") != "complete" or x.get("protocol") != PROTOCOL
               or x.get("model_digest") != DIGEST or x.get("n") != 2
               or len(x.get("cases", [])) != 2 for x in parts)
        or len({x["source_sha"] for x in parts}) != 1):
        raise RuntimeError("Incomplete or inconsistent original physical source shards")
    cases = [r for x in sorted(parts, key=lambda x:x["shard"]) for r in x["cases"]]
    if len(cases) != N or len({r["id"] for r in cases}) != N:
        raise RuntimeError("Missing or duplicate source questions")
    scores = {}
    for arm in ARMS:
        scores[arm] = {
            k: sum(int(r["arms"][arm][k]) for r in cases)
            for k in ("raw_exact", "locked_lastline_exact", "all_two_supports_read",
                      "completed", "tool_calls", "tool_errors", "duplicate_calls")
        }
        scores[arm]["seconds"] = round(
            sum(r["arms"][arm]["seconds"] for r in cases), 1)
    def paired(field):
        rescue = [r["id"] for r in cases if not r["arms"]["native"][field]
                  and r["arms"]["linked"][field]]
        regress = [r["id"] for r in cases if r["arms"]["native"][field]
                   and not r["arms"]["linked"][field]]
        return {"rescues": rescue, "regressions": regress,
                "net": len(rescue) - len(regress)}
    raw, parsed = paired("raw_exact"), paired("locked_lastline_exact")
    n = scores["native"]
    l = scores["linked"]
    ratio = l["tool_calls"] / max(1, n["tool_calls"])
    gate = (
        parsed["net"] >= GATE["net_end_to_end"] and
        len(parsed["regressions"]) <= GATE["regressions"] and
        l["all_two_supports_read"] - n["all_two_supports_read"] >= GATE["net_evidence"] and
        ratio <= GATE["max_call_ratio"]
    )
    return {
        "status": "COMPLETE_HOTPOT_SOURCE_SHIFT_NOT_PROMOTED",
        "protocol": PROTOCOL, "source_sha": parts[0]["source_sha"],
        "n": N, "dataset": REPO, "revision": REV,
        "same_Qwen9B_digest": DIGEST,
        "arms": scores, "raw_link_effect": raw,
        "locked_parser_link_effect": parsed,
        "parser_rescues_native": sum(
            not r["arms"]["native"]["raw_exact"] and
            r["arms"]["native"]["locked_lastline_exact"] for r in cases),
        "parser_regressions_native": sum(
            r["arms"]["native"]["raw_exact"] and
            not r["arms"]["native"]["locked_lastline_exact"] for r in cases),
        "parser_rescues_linked": sum(
            not r["arms"]["linked"]["raw_exact"] and
            r["arms"]["linked"]["locked_lastline_exact"] for r in cases),
        "parser_regressions_linked": sum(
            r["arms"]["linked"]["raw_exact"] and
            not r["arms"]["linked"]["locked_lastline_exact"] for r in cases),
        "tool_calls_ratio": round(ratio, 3), "predeclared_gate": GATE,
        "positive_screen_only": gate, "promotion_allowed": False,
        "no_GAIA_tasks": True, "no_adapter_training": True,
        "claim_boundary": "HotpotQA public distractor validation source-shift, not GAIA, not hidden nor independent of possible model pretraining. Parser frozen before any model inference; no gold in prompts or tool observations. Comparison is under GAIA-like exact scorer, not official Hotpot leaderboard metric. Small sample, no GAIA score gain.",
        "cases": cases,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", type=Path)
    p.add_argument("--shard", type=int)
    p.add_argument("--inputs", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--work", type=Path, default=Path("hotpot-work"))
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.inputs:
        result = aggregate([json.loads(x.read_text()) for x in a.inputs])
        print("HOTPOT_SOURCE_SHIFT_FINAL", json.dumps(
            {k: v for k, v in result.items() if k != "cases"}, sort_keys=True), flush=True)
    elif a.cohort:
        if a.shard is None:
            p.error("--shard required for inference")
        result = run_shard(json.loads(a.cohort.read_text()), a.shard, a.work)
    else:
        from datasets import load_dataset
        source = list(islice(load_dataset(
            REPO, "distractor", split="validation", revision=REV,
            streaming=True), 800))
        result = freeze(source)
        print("HOTPOT_SOURCE_FROZEN", result["cases_sha256"], flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
