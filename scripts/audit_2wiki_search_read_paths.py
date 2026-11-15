"""PRO source-feasibility screen for genuine non-oracle search+read decisions.

Original 2Wiki TRAIN context has source text and annotated supporting spans.
The simulated local full-text title index is constructed from ALL visible
per-task documents, INCLUDING distractors; no support annotations affect
action choices. Gold labels are used only AFTER search/read decisions.

This is a source-environment screen, not model-generated or verified teacher
actions and not a substitute for a native-agent closed-loop evaluation.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from itertools import islice
from pathlib import Path
import re
from collections import Counter

from gaia_small_agent.agent.loop import MAX_TOOL_OBSERVATION_CHARS


def norm(s: str) -> str:
    return " ".join(s.casefold().split())


def visible(title: str, seen: str) -> bool:
    title = norm(title)
    return len(title) >= 4 and re.search(
        r"(?<!\w)" + re.escape(title) + r"(?!\w)", norm(seen)
    ) is not None


def context_support(row: dict) -> tuple[dict[str, str], dict[str, list[str]]]:
    raw = row["context"]
    if isinstance(raw, dict):
        originals = dict(zip(raw["title"], raw["sentences"], strict=True))
    else:
        originals = dict(raw)
    context = {k: " ".join(v) for k, v in originals.items()}
    support = {}
    raw_support = row["supporting_facts"]
    facts = (zip(raw_support["title"], raw_support["sent_id"], strict=True)
             if isinstance(raw_support, dict) else raw_support)
    for title, sent_id in facts:
        if title not in originals or not 0 <= sent_id < len(originals[title]):
            raise ValueError("invalid published evidence pointer")
        support.setdefault(title, []).append(originals[title][sent_id])
    return context, support


def replay(row: dict, limit: int = 12) -> dict:
    """Choose which title to search using only the question and prior reads."""
    pages, support = context_support(row)
    seen = str(row["question"])
    visited: list[str] = []
    actions = []
    for step in range(limit):
        # The index enumerates all publisher-provided documents, including
        # distractors. Support labels are never consulted by this selector.
        choices = [title for title in pages
                   if title not in visited and visible(title, seen)]
        if not choices:
            break
        title = sorted(choices, key=lambda t: sha256(t.encode()).hexdigest())[0]
        path = sha256(title.encode()).hexdigest()[:12] + ".txt"
        snippet = pages[title][:120]
        actions.append({"kind": "search", "query": title,
                        "result_path": path})
        seen += "\n" + snippet
        delivered = pages[title][:MAX_TOOL_OBSERVATION_CHARS]
        actions.append({"kind": "read", "source": path,
                        "new_observation": len(delivered) > len(snippet)})
        seen += "\n" + delivered
        visited.append(title)
    delivered_support = sum(
        title in visited and all(sentence in pages[title][:MAX_TOOL_OBSERVATION_CHARS]
                                 for sentence in sentences)
        for title, sentences in support.items()
    )
    has_all = delivered_support == len(support)
    useful = all(a["new_observation"] for a in actions if a["kind"] == "read")
    long_path = (
        len(support) >= 4 and has_all and len(actions) >= 8
        and useful and len(visited) <= 2 * len(support)
    )
    return {
        "support_documents": len(support),
        "total_documents": len(pages),
        "visited": len(visited),
        "delivered_support": delivered_support,
        "complete": has_all,
        "nonredundant_search_reads": len(actions) if useful else 0,
        "long_candidate": long_path,
        "action_names": [a["kind"] for a in actions],
        "oracle_actions_used": False,
    }


def audit(rows: list[dict], revision: str) -> dict:
    if not rows:
        raise ValueError("empty source")
    ids = set()
    results = []
    for row in rows:
        identity = str(row.get("id") or row.get("_id") or "")
        if not identity or identity in ids:
            raise ValueError("duplicate or missing publisher identity")
        ids.add(identity)
        results.append(replay(row))
    return {
        "protocol": "2wiki-source-read-search-nonoracle-v1",
        "original_source": "framolfese/2WikiMultihopQA",
        "pinned_revision": revision, "split": "train",
        "source_selection": f"first {len(rows)} publisher TRAIN rows (not random)",
        "rows": len(rows),
        "support_distribution": dict(sorted(Counter(x["support_documents"] for x in results).items())),
        "all_support_delivered": sum(x["complete"] for x in results),
        "four_document_complete": sum(x["complete"] and x["support_documents"] >= 4 for x in results),
        "real_content_search_read_long_corpus_candidates": sum(x["long_candidate"] for x in results),
        "source_read_action_count_distribution": dict(sorted(Counter(x["nonredundant_search_reads"] for x in results).items())),
        "model_agent_actions_tested": 0,
        "training_labels_exported": 0,
        "training_ready": False,
        "caveat": (
            "Full-text per-task title index includes distractors, but is a simulated local index. "
            "This is only path feasibility, not teacher action quality or closed-loop Qwen success; "
            "public source may appear in pretraining."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=1024)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not 1 <= args.limit <= 5000 or args.output.exists():
        raise ValueError("invalid output or limit")
    from datasets import load_dataset
    from huggingface_hub import HfApi
    repo = "framolfese/2WikiMultihopQA"
    revision = HfApi().dataset_info(repo).sha
    rows = list(islice(load_dataset(repo, split="train",
                                   streaming=True, revision=revision), args.limit))
    if len(rows) != args.limit:
        raise ValueError("unexpected source truncation")
    result = audit(rows, revision)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("2WIKI_NONORACLE_PATHS", json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
