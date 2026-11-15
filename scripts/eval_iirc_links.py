"""Test source-provided link positions in an autonomous Qwen tool loop."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from html import unescape
from io import TextIOWrapper
import json
from pathlib import Path
import re
import tarfile
from time import monotonic
from urllib.request import urlopen

from gaia_small_agent.agent.loop import AgentRuntime
from gaia_small_agent.benchmark.scoring import gaia_score
from gaia_small_agent.model.ollama import OllamaModel
from gaia_small_agent.tools import ReadTool
from gaia_small_agent.training.trajectories import _materialize_files


MODEL = "qwen3.5:9b-q4_K_M"
DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
BASE_URL = "http://127.0.0.1:11434"
PROTOCOL = "iirc-dev-source-links-8-v1"
ARMS = ("flat", "linked")
N_SHARDS = 4
ANCHOR = re.compile(r" \[pages/p\d+\.txt\]")
HTML = re.compile(r"<[^>]*>")
VISIBLE_PREFIX = 1600


def key(text: str) -> str:
    return " ".join(unescape(text).casefold().split())


def rank(text: str) -> str:
    return sha256(f"{PROTOCOL}:{text}".encode()).hexdigest()


def load_split(archive: Path, split: str) -> list[dict]:
    with tarfile.open(archive, "r:gz") as source:
        return json.load(source.extractfile(f"iirc_train_dev/{split}.json"))


def load_articles(archive: Path, wanted: set[str]) -> dict[str, str]:
    """Read original objects from the publisher's line-formatted JSON."""
    selected = {}
    decoder = json.JSONDecoder()
    with tarfile.open(archive, "r:gz") as source:
        with TextIOWrapper(source.extractfile("context_articles.json"), encoding="utf-8") as lines:
            for line in lines:
                line = line.strip().removesuffix(",")
                if line in {"{", "}"}:
                    continue
                title, end = decoder.raw_decode(line)
                name = key(title)
                if name in wanted and name not in selected:
                    body = json.loads(line[end:].lstrip().removeprefix(":").lstrip())
                    selected[name] = unescape(HTML.sub("", body))
                if wanted <= selected.keys():
                    break
    return selected


def workspace(page: dict, articles: dict[str, str]) -> tuple[dict, dict[str, str], str]:
    paths = {}
    for link in page["links"]:
        name = key(link["target"])
        if name in articles and name not in paths:
            paths[name] = f"pages/p{len(paths):02}.txt"
    names = {key(link["target"]): link["target"] for link in page["links"]}
    files = {path: articles[name] for name, path in paths.items()}
    files["index.txt"] = "\n".join(f"{path} | {names[name]}" for name, path in paths.items())
    text = page["text"]
    files["main.txt"] = text
    insertions = {}
    for link in page["links"]:
        name = key(link["target"])
        start, end = link["indices"]
        if name in paths and 0 <= start < end <= len(text) and text[start:end].strip():
            insertions[end] = paths[name]
    linked = text
    for offset, path in sorted(insertions.items(), reverse=True):
        linked = linked[:offset] + f" [{path}]" + linked[offset:]
    return files, paths, linked


def published_support(row: dict, paths: dict[str, str]) -> dict[str, list[str]] | None:
    groups = defaultdict(list)
    for span in row["context"]:
        name = key(span["passage"])
        path = "main.txt" if name == "main" else paths.get(name)
        if path is None:
            return None
        snippet = key(span["text"])
        if snippet:
            groups[path].append(snippet)
    if "main.txt" not in groups or len(groups) not in {3, 4}:
        return None
    return {
        path: sorted({span for span in spans if not any(span != other and span in other for other in spans)})
        for path, spans in groups.items()
    }


def all_visible(spans: list[str], text: str) -> bool:
    observation = key(ANCHOR.sub("", text))
    return bool(spans) and all(span in observation for span in spans)


def freeze(dev: list[dict], articles: dict[str, str], forbidden: set[str]) -> list[dict]:
    cases = []
    seen_support = set()
    for page in sorted(dev, key=lambda page: rank(page["pid"])):
        if key(page["title"]) in forbidden or not 6 <= len(page["links"]) <= 30:
            continue
        files, paths, linked = workspace(page, articles)
        if not 6 <= len(paths) <= 30 or linked == files["main.txt"]:
            continue
        for row in sorted(page["questions"], key=lambda row: rank(row["qid"])):
            span = row["answer"].get("answer_spans", [])
            if row["answer"]["type"] != "span" or len(span) != 1 or not span[0]["text"].strip():
                continue
            support = published_support(row, paths)
            if not support or not all(all_visible(parts, files[path][:VISIBLE_PREFIX])
                                      for path, parts in support.items()):
                continue
            if not all_visible(support["main.txt"], linked[:VISIBLE_PREFIX]):
                continue
            titles = {name for name, path in paths.items() if path in support}
            if titles & (forbidden | seen_support) or any(
                f" [{paths[title]}]" not in linked[:VISIBLE_PREFIX] for title in titles
            ):
                continue
            seen_support.update(titles)
            cases.append({
                "id": row["qid"], "page": page["pid"], "page_title": key(page["title"]),
                "question": row["question"], "files": files, "linked_main": linked,
                "answer": span[0]["text"].strip(), "support": support, "support_titles": sorted(titles),
            })
            break
        if len(cases) == N_SHARDS * 2:
            return cases
    raise ValueError(f"did not find 8 source-verified independent multi-document tasks ({len(cases)})")


def delivered_support(trace, support: dict[str, list[str]]) -> dict:
    calls = {}
    read = defaultdict(list)
    for entry in trace:
        event = entry.data
        if entry.kind == "tool_call" and event.get("name") == "read":
            source = event["arguments"].get("source")
            if isinstance(source, str):
                calls[event["id"]] = Path(source).as_posix()
        elif entry.kind == "tool_result" and event.get("ok"):
            path = calls.get(event["id"])
            if path in support:
                content = event["content"]
                if content.startswith("ToolMeta: "):
                    content = content.partition("\n")[2]
                limit = event.get("metadata", {}).get("returned_chars", len(content))
                read[path].append(content[:limit])
    delivered = {
        path: sum(any(all_visible([part], obs) for obs in read[path]) for part in spans)
        for path, spans in support.items()
    }
    return {
        "delivered_docs": sorted(path for path, count in delivered.items() if count == len(support[path])),
        "annotated_docs": len(support),
        "delivered_spans": sum(delivered.values()),
        "annotated_spans": sum(map(len, support.values())),
    }


def run_arm(model, case: dict, root: Path, arm: str) -> dict:
    files = dict(case["files"])
    if arm == "linked":
        files["main.txt"] = case["linked_main"]
    _materialize_files({"files": files}, root)
    prompt = (
        "Answer the question using the local documents. The main article is main.txt. "
        "The directory of related articles is index.txt.\n\n"
        f"Question: {case['question']}"
    )
    started = monotonic()
    result = AgentRuntime(model, [ReadTool()], max_steps=8).run(prompt, root)
    support = delivered_support(result.trace, case["support"])
    return {
        "answer": result.answer, "exact": bool(result.completed and gaia_score(result.answer, case["answer"])),
        "completed": result.completed, "stop": result.stop_reason, "tool_calls": result.metrics.tool_calls,
        "tool_errors": result.metrics.tool_errors, "seconds": round(monotonic() - started, 1),
        **support, "full_support": len(support["delivered_docs"]) == support["annotated_docs"],
    }


def run_shard(cohort: dict, shard: int, root: Path) -> dict:
    with urlopen(BASE_URL + "/api/tags", timeout=10) as response:
        models = json.load(response)["models"]
    if not any(row["name"] == MODEL and row["digest"] == DIGEST for row in models):
        raise ValueError("physical Qwen model changed")
    model = OllamaModel(model=MODEL, base_url=BASE_URL, timeout_s=400, max_new_tokens=512)
    cases = []
    for offset, case in enumerate(cohort["cases"][2 * shard:2 * shard + 2]):
        order = ARMS if (shard + offset) % 2 == 0 else ARMS[::-1]
        outcome = {arm: run_arm(model, case, root / arm / case["id"], arm) for arm in order}
        cases.append({
            "id": case["id"], "page": case["page"], "reference": case["answer"],
            "support_doc_count": len(case["support"]), **outcome,
        })
        print("IIRC_QWEN_PAIR", case["id"], {arm: (x["exact"], x["full_support"]) for arm, x in outcome.items()}, flush=True)
    return {"protocol": PROTOCOL, "digest": DIGEST, "shard": shard, "cohort_sha": cohort["cohort_sha"], "cases": cases}


def aggregate(shards: list[dict]) -> dict:
    if sorted(item["shard"] for item in shards) != list(range(N_SHARDS)) or any(
        item["protocol"] != PROTOCOL or item["digest"] != DIGEST or len(item["cases"]) != 2 for item in shards
    ) or len({item["cohort_sha"] for item in shards}) != 1:
        raise ValueError("missing or inconsistent paired shards")
    cases = sorted((row for item in shards for row in item["cases"]), key=lambda row: row["id"])
    if len({row["id"] for row in cases}) != 8 or len({row["page"] for row in cases}) != 8:
        raise ValueError("repeated page or task")
    summary = {
        arm: {
            key: sum(row[arm][key] for row in cases)
            for key in ("exact", "completed", "tool_calls", "tool_errors", "full_support",
                        "delivered_spans", "annotated_spans")
        }
        for arm in ARMS
    }
    for arm in ARMS:
        summary[arm]["seconds"] = round(sum(row[arm]["seconds"] for row in cases), 1)
        summary[arm]["wrong_with_full_support"] = sum(
            not row[arm]["exact"] and row[arm]["full_support"] for row in cases
        )
    rescues = [row["id"] for row in cases if not row["flat"]["exact"] and row["linked"]["exact"]]
    regressions = [row["id"] for row in cases if row["flat"]["exact"] and not row["linked"]["exact"]]
    gains = summary["linked"]["full_support"] - summary["flat"]["full_support"]
    return {
        "protocol": PROTOCOL, "model": MODEL, "digest": DIGEST, "tasks": len(cases),
        "cohort_sha": shards[0]["cohort_sha"], "arms": summary, "rescues": rescues,
        "regressions": regressions, "positive_screen": len(rescues) >= 2 and not regressions and gains >= 2
        and summary["linked"]["completed"] >= summary["flat"]["completed"],
        "promotion_allowed": False, "cases": cases,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path)
    p.add_argument("--articles", type=Path)
    p.add_argument("--cohort", type=Path)
    p.add_argument("--work", type=Path, default=Path("iirc-work"))
    p.add_argument("--shard", type=int, choices=range(N_SHARDS))
    p.add_argument("--inputs", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.inputs:
        data = aggregate([json.loads(path.read_text()) for path in args.inputs])
        print("IIRC_QWEN9B_LINKS_FINAL", json.dumps({k: v for k, v in data.items() if k != "cases"}), flush=True)
    elif args.cohort:
        cohort = json.loads(args.cohort.read_text())
        if args.shard is None:
            p.error("--shard is required with --cohort")
        if sha256(json.dumps(cohort["cases"], sort_keys=True).encode()).hexdigest() != cohort["cohort_sha"]:
            raise ValueError("frozen IIRC workspace and label bytes changed")
        data = run_shard(cohort, args.shard, args.work)
    else:
        if args.source is None or args.articles is None:
            p.error("--source and --articles are required to freeze a cohort")
        dev = load_split(args.source, "dev")
        train = load_split(args.source, "train")
        forbidden = {key(page["title"]) for page in train}
        if {page["pid"] for page in dev} & {page["pid"] for page in train}:
            raise ValueError("IIRC train/dev primary page IDs overlap")
        needed = {key(link["target"]) for page in dev for link in page["links"]}
        articles = load_articles(args.articles, needed)
        cases = freeze(dev, articles, forbidden)
        data = {"cohort_sha": sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(), "cases": cases}
        print("IIRC_FROZEN", [(x["id"], x["page"], len(x["support"])) for x in cases], flush=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
