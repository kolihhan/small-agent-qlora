"""Paired next-action and terminal task test on physical news evidence states."""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
from time import monotonic

import requests

from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.agent.types import AssistantTurn, ModelRuntimeError, ToolCall
from gaia_small_agent.benchmark.scoring import gaia_score
from gaia_small_agent.model.ollama import OllamaModel
from gaia_small_agent.tools import ReadTool
from gaia_small_agent.tools.base import ToolResult
from gaia_small_agent.training.trajectories import _materialize_files


PROTOCOL = "multihoprag-evidence-state-actions-8-v1"
MODEL = "qwen3.5:9b-q4_K_M"
DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
BASE = "http://127.0.0.1:11434"
ARMS = ("native", "typed")
STATES = ("missing", "complete")
MAX_TOKENS = 320


def rank(s: str) -> str:
    return sha256(f"{PROTOCOL}:{s}".encode()).hexdigest()


def plain(s: str) -> str:
    return " ".join(s.split())


def has(value: str, text: str) -> bool:
    return bool(re.search(r"(?<!\w)" + re.escape(value.casefold()) + r"(?!\w)", plain(text).casefold()))


def slice_support(article: dict, facts: list[str]) -> str | None:
    """A bounded contiguous copy of the publisher's actual article."""
    body = plain(article["body"])
    matches = [re.search(re.escape(fact), body, re.IGNORECASE) for fact in facts]
    if any(match is None for match in matches):
        return None
    start = min(match.start() for match in matches)
    end = max(match.end() for match in matches)
    if end - start > 900:
        return None
    return body[max(0, start - 75):min(len(body), end + 75)]


def freeze(questions: list[dict], corpus: list[dict]) -> list[dict]:
    articles = {a["url"]: a for a in corpus}
    used = set()
    used_answers = set()
    cases = []
    count: dict[int, int] = defaultdict(int)
    for row in sorted(questions, key=lambda r: (
        -len({item["url"] for item in r["evidence_list"]}), rank(r["query"]),
    )):
        answer = plain(str(row["answer"]))
        if answer.casefold() in {"yes", "no", "insufficient information"} or not 3 <= len(answer) <= 64:
            continue
        if answer.casefold() in used_answers or has(answer, row["query"]):
            continue
        facts = defaultdict(list)
        for item in row["evidence_list"]:
            facts[item["url"]].append(plain(item["fact"]))
        n = len(facts)
        if n not in (3, 4) or count[n] == 4 or used.intersection(facts) or any(url not in articles for url in facts):
            continue
        supported = {url: slice_support(articles[url], texts) for url, texts in facts.items()}
        if any(not text for text in supported.values()):
            continue
        decisive = [url for url, text in supported.items() if has(answer, text)]
        if len(decisive) != 1:
            continue
        words = set(re.findall(r"[a-z]{4,}", row["query"].casefold()))
        other = [url for url, a in articles.items() if url not in facts and url not in used
                 and not has(answer, a["title"]) and not has(answer, a["body"][:1600])
                 and len(plain(a["body"])) >= len(supported[decisive[0]])
                 and a["category"] == articles[decisive[0]]["category"]]
        other.sort(key=lambda url: (-len(words.intersection(re.findall(r"[a-z]{4,}",
                   (articles[url]["title"] + " " + articles[url]["body"][:800]).casefold()))), rank(url)))
        if len(other) < 6 - n:
            continue
        distractors = other[:6 - n]
        urls = sorted([*supported, *distractors], key=rank)
        paths = {url: f"docs/d{i+1}.txt" for i, url in enumerate(urls)}
        files = {paths[url]: supported[url] if url in supported else plain(articles[url]["body"])[:850]
                 for url in urls}
        files[paths[distractors[0]]] = plain(articles[distractors[0]]["body"])[:len(supported[decisive[0]])]
        inventory = "\n".join(
            f"{paths[url]} | {articles[url]['source']} | {articles[url]['published_at'][:10]}"
            for url in urls
        )
        if has(answer, inventory):
            continue
        needed = paths[decisive[0]]
        required = sorted(paths[url] for url in supported)
        warm_missing = sorted([path for path in required if path != needed] + [paths[distractors[0]]])
        cases.append({
            "id": rank(row["query"])[:16], "question": row["query"], "answer": answer,
            "question_type": row["question_type"], "evidence_documents": n, "inventory": inventory,
            "files": files, "required": required, "needed": needed,
            "warm": {"complete": required, "missing": warm_missing},
        })
        used.update(supported)
        used_answers.add(answer.casefold())
        count[n] += 1
        if len(cases) == 8:
            return cases
    raise ValueError(f"unable to freeze independent real source cases: {dict(count)}")


class LocalRead(ReadTool):
    def __init__(self, paths: set[str]):
        self.paths = paths

    def run(self, arguments, workspace):
        source = Path(arguments["source"]).as_posix()
        if source not in self.paths:
            return ToolResult(False, "Choose one of the listed local files.", "UNKNOWN_SOURCE")
        return super().run({**arguments, "source": source}, workspace)


class SeededModel:
    def __init__(self, prefix: list[str], arm: str):
        self.prefix = prefix
        self.arm = arm
        self.native = OllamaModel(model=MODEL, base_url=BASE, timeout_s=270, max_new_tokens=MAX_TOKENS)
        self.calls = 0

    def complete(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        if self.prefix:
            prefix, self.prefix = self.prefix, []
            return AssistantTurn("", [ToolCall(f"seed-{i}", "read", {"source": path})
                                      for i, path in enumerate(prefix)])
        self.calls += 1
        if self.arm == "native":
            return self.native.complete(messages, tools)
        grammar = {
            "type": "object", "additionalProperties": False, "required": ["action", "value"],
            "properties": {
                "action": {"type": "string", "enum": ["read", "submit"] if tools else ["submit"]},
                "value": {"type": "string"},
            },
        }
        system = dict(messages[0])
        system["content"] += (
            '\nRespond with JSON using action and value. For action "read", value is the listed file path. '
            'For action "submit", value is the requested answer itself. '
            'Choose read while evidence is missing and the read action is available.'
        )
        if not tools:
            system["content"] += ' The read budget has ended. Use submit.'
        try:
            response = requests.post(BASE + "/api/chat", json={
                "model": MODEL, "stream": False, "think": False, "format": grammar,
                "messages": [system, *messages[1:]],
                "options": {"temperature": 0, "num_predict": MAX_TOKENS},
            }, timeout=270)
            response.raise_for_status()
            choice = json.loads(response.json()["message"]["content"])
            if not isinstance(choice.get("value"), str) or choice.get("action") not in grammar["properties"]["action"]["enum"]:
                raise ValueError("response does not match action grammar")
        except (requests.RequestException, ValueError, KeyError, AttributeError):
            raise ModelRuntimeError("model_error", "invalid constrained action response") from None
        if choice["action"] == "read":
            return AssistantTurn("", [ToolCall(f"typed-{self.calls}", "read", {"source": choice["value"]})])
        return AssistantTurn(choice["value"].strip())


def run_arm(case: dict, state: str, arm: str, root: Path) -> dict:
    _materialize_files({"files": case["files"]}, root)
    model = SeededModel(case["warm"][state], arm)
    runtime = AgentRuntime(model, [LocalRead(set(case["files"]))], max_steps=4)
    prompt = (
        "Answer using the local news sources. Continue reading if evidence is incomplete.\n\n"
        f"Question: {case['question']}\n\nAvailable local files:\n{case['inventory']}"
    )
    started = monotonic()
    run = runtime.run(prompt, root)
    paths = {}
    seed_returns = set()
    real_returns = set()
    first_reads = set()
    for event in run.trace:
        data = event.data
        if event.kind == "tool_call":
            paths[data["id"]] = Path(data["arguments"].get("source", "")).as_posix()
        if event.kind == "tool_result" and data.get("ok"):
            path = paths[data["id"]]
            chars = data.get("metadata", {}).get("returned_chars", 0)
            if path in case["files"] and chars >= len(case["files"][path]):
                (seed_returns if event.step == 1 else real_returns).add(path)
                if event.step == 2:
                    first_reads.add(path)
    if seed_returns != set(case["warm"][state]):
        raise ValueError("a seeded state did not deliver the frozen physical evidence")
    sufficient = set(case["required"]) <= seed_returns | real_returns
    exact = bool(run.completed and gaia_score(run.answer, case["answer"]))
    return {
        "exact": exact, "supported_exact": bool(exact and sufficient), "completed": run.completed,
        "first_read_needed": case["needed"] in first_reads, "first_submit": run.completed and run.metrics.steps == 2,
        "premature_submit": run.completed and not sufficient, "model_calls": model.calls,
        "additional_reads": run.metrics.tool_calls - len(seed_returns),
        "tool_errors": run.metrics.tool_errors, "stop": run.stop_reason,
        "seconds": round(monotonic() - started, 2),
    }


def run_shard(cases: list[dict], shard: int, root: Path) -> dict:
    tags = requests.get(BASE + "/api/tags", timeout=10)
    tags.raise_for_status()
    if not any(x["name"] == MODEL and x["digest"] == DIGEST for x in tags.json()["models"]):
        raise ValueError("unexpected Qwen model")
    selections = []
    for j, case in enumerate(cases[2 * shard:2 * shard + 2]):
        outcomes = {}
        for state in STATES:
            order = ARMS if (shard + j + (state == "complete")) % 2 == 0 else ARMS[::-1]
            outcomes[state] = {arm: run_arm(case, state, arm, root / case["id"] / state / arm) for arm in order}
        selections.append({"id": case["id"], "evidence_documents": case["evidence_documents"], "outcomes": outcomes})
        print("EVIDENCE_STATE_CASE", case["id"], json.dumps(outcomes), flush=True)
    return {"protocol": PROTOCOL, "digest": DIGEST, "shard": shard, "cases": selections,
            "cohort_id": rank("|".join(case["id"] for case in cases))}


def aggregate(shards: list[dict]) -> dict:
    if sorted(s["shard"] for s in shards) != list(range(4)) or len({s["cohort_id"] for s in shards}) != 1 or any(
        s["digest"] != DIGEST or s["protocol"] != PROTOCOL or len(s["cases"]) != 2 for s in shards
    ):
        raise ValueError("incomplete paired shards")
    cases = [case for s in sorted(shards, key=lambda s: s["shard"]) for case in s["cases"]]
    if len({case["id"] for case in cases}) != 8:
        raise ValueError("duplicate tasks")
    metrics = ("exact", "supported_exact", "completed", "first_read_needed", "first_submit",
               "premature_submit", "model_calls", "additional_reads", "tool_errors", "seconds")
    totals = {
        state: {
            arm: {key: round(sum(case["outcomes"][state][arm][key] for case in cases), 2) for key in metrics}
            for arm in ARMS
        }
        for state in STATES
    }
    wins, losses = [], []
    for state in STATES:
        for case in cases:
            row = case["outcomes"][state]
            pair = (row["native"]["supported_exact"], row["typed"]["supported_exact"])
            if pair == (False, True):
                wins.append((state, case["id"]))
            if pair == (True, False):
                losses.append((state, case["id"]))
    passed = (len(wins) - len(losses) >= 3 and len(losses) <= 1 and
              {state for state, _ in wins} == set(STATES) and
              all(totals[s]["typed"]["completed"] >= totals[s]["native"]["completed"] for s in STATES) and
              totals["missing"]["typed"]["first_read_needed"] >= totals["missing"]["native"]["first_read_needed"] and
              totals["missing"]["typed"]["premature_submit"] <= totals["missing"]["native"]["premature_submit"])
    return {"protocol": PROTOCOL, "digest": DIGEST, "cohort_id": shards[0]["cohort_id"],
            "tasks": len(cases), "seeded_states": 2 * len(cases), "metrics": totals,
            "rescues": wins, "regressions": losses, "positive_screen": passed,
            "promotion_allowed": False, "cases": cases}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--questions", type=Path)
    p.add_argument("--corpus", type=Path)
    p.add_argument("--shard", type=int, choices=range(4))
    p.add_argument("--inputs", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.inputs:
        result = aggregate([json.loads(f.read_text()) for f in a.inputs])
        print("EVIDENCE_STATE_FINAL", json.dumps({k: v for k, v in result.items() if k != "cases"}), flush=True)
    elif a.questions and a.corpus:
        cases = freeze(json.loads(a.questions.read_text()), json.loads(a.corpus.read_text()))
        print("EVIDENCE_STATE_FROZEN", [(x["id"], x["evidence_documents"]) for x in cases], flush=True)
        if a.shard is None:
            return
        result = run_shard(cases, a.shard, a.output.parent / "state-work")
    else:
        p.error("use --inputs or provide both --questions and --corpus")
    with a.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, sort_keys=True)
        out.write("\n")


if __name__ == "__main__":
    main()
