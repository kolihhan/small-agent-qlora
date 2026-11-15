"""PRO source-feasibility audit: independently sourced 2Wiki TRAIN decisions.

This only checks whether authentic multi-hop evidence may support longer
read policies. Never exports actions, training examples, or reference answers.
Gold support labels are used for post-hoc feasibility, not as state-visible
oracle hints to an Agent.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import re
from pathlib import Path

from gaia_small_agent.agent.loop import MAX_TOOL_OBSERVATION_CHARS


def norm(text: str) -> str:
    return " ".join(text.casefold().split())


def literal_mentioned(text: str, title: str) -> bool:
    candidate = norm(title)
    return len(candidate) > 2 and bool(
        re.search(r"(?<!\w)" + re.escape(candidate) + r"(?!\w)", norm(text))
    )


def canonical_row(row: dict) -> tuple[dict[str, str], dict[str, list[str]]]:
    raw = row["context"]
    if isinstance(raw, dict):
        titles = raw["title"]
        paragraphs = raw.get("sentences") or raw.get("content")
        if paragraphs is None:
            raise ValueError("missing context sentences")
        context = dict(zip(titles, [" ".join(s) for s in paragraphs], strict=True))
    else:
        context = {title: " ".join(sentences) for title, sentences in raw}
    raw_support = row["supporting_facts"]
    if isinstance(raw_support, dict):
        pairs = zip(raw_support["title"], raw_support["sent_id"], strict=True)
    else:
        pairs = raw_support
    support: dict[str, list[str]] = {}
    for title, idx in pairs:
        if title not in context or not isinstance(idx, int) or idx < 0:
            raise ValueError("invalid supporting paragraph reference")
        # Resolve sentences from original source, never generated text.
        if isinstance(raw, dict):
            original = dict(zip(raw["title"], raw.get("sentences") or raw["content"], strict=True))
        else:
            original = dict(raw)
        sentences = original[title]
        if idx >= len(sentences):
            raise ValueError("supporting sentence index outside source")
        support.setdefault(title, []).append(sentences[idx])
    return context, support


def inspect_row(row: dict) -> dict:
    context, support = canonical_row(row)
    question = row["question"]
    if len(support) < 2:
        return {"support_documents": len(support), "reachable": 0,
                "all_support_visible": False, "long_candidate": False}
    # An index contains ALL provided documents, including distractors.
    index = [(sha256(title.encode()).hexdigest()[:12], title) for title in context]
    seen = question
    unread = set(support)
    visited: list[str] = []
    while unread:
        available = sorted(t for t in unread if literal_mentioned(seen, t))
        if not available:
            break
        chosen = available[0]  # deterministic reachability, NOT oracle next-action supervision
        article = context[chosen]
        actual_observation = article[:MAX_TOOL_OBSERVATION_CHARS]
        visited.append(chosen)
        seen += "\n" + actual_observation
        unread.remove(chosen)
    # Observed support sentence must be ENTIRELY in bounded message.
    delivered = sum(all(x in context[t][:MAX_TOOL_OBSERVATION_CHARS] for x in support[t])
                    for t in visited)
    return {
        "support_documents": len(support),
        "reachable": len(visited),
        "all_support_visible": len(visited) == len(support) and delivered == len(support),
        "delivered_support_documents": delivered,
        "long_candidate": len(support) >= 6 and delivered >= 6,
        "index_documents": len(index),
    }


def audit(rows: list[dict], revision: str) -> dict:
    if not rows:
        raise ValueError("nonempty original TRAIN sample required")
    seen = set()
    summaries = []
    invalid = []
    for i, row in enumerate(rows):
        identity = str(row.get("id") or row.get("_id") or "")
        if not identity or identity in seen:
            raise ValueError("missing or duplicated source identity")
        seen.add(identity)
        try:
            summaries.append(inspect_row(row))
        except (ValueError, TypeError, KeyError) as error:
            invalid.append({"identity_hash": sha256(identity.encode()).hexdigest()[:12],
                            "error_type": type(error).__name__})
    counts = Counter(s["support_documents"] for s in summaries)
    eligible = sum(s["long_candidate"] for s in summaries)
    return {
        "protocol": "2wiki-train-visible-evidence-feasibility-v1",
        "dataset": "framolfese/2WikiMultihopQA", "source_revision": revision,
        "split": "train", "selection": f"source-order first {len(rows)} (not representative random sampling)",
        "sampled": len(rows), "verified_cases": len(summaries),
        "invalid_cases": invalid, "support_documents_distribution": dict(sorted(counts.items())),
        "reach_all_support": sum(s["all_support_visible"] for s in summaries),
        "literal_reachable_support_docs": sum(s["reachable"] for s in summaries),
        "actual_delivered_support_docs": sum(s["delivered_support_documents"] for s in summaries),
        "long_ge6_distinct_support_docs_and_visible": eligible,
        "training_labels_exported": 0,
        "training_ready": False,
        "decision": "SOURCE_INSUFFICIENT" if eligible == 0 else "NEEDS_ACTION_QUALITY_AUDIT",
        "caveat": "Literal-title conservative reachability is NOT a unique correct next-action label. Annotated docs alone are not verified seven-step agent trajectories. Public source may be in model pretraining.",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=1024)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not 1 <= args.limit <= 5000 or args.output.exists():
        raise ValueError("invalid row limit or output exists")
    from datasets import load_dataset
    from huggingface_hub import HfApi
    repo = "framolfese/2WikiMultihopQA"
    revision = HfApi().dataset_info(repo).sha
    stream = load_dataset(repo, split="train", streaming=True, revision=revision)
    from itertools import islice
    rows = list(islice(stream, args.limit))
    if len(rows) != args.limit:
        raise ValueError("source has fewer rows than requested")
    report = audit(rows, revision)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("2WIKI_SOURCE_AUDIT", json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
