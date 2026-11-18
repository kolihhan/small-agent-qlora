"""Create TRAIN-only 3/4-hop MuSiQue source tasks; never synthesize teacher actions.

The release is CC-BY-4.0. This is candidate conversion, NOT verified
agent behavior, a GPU training set or evidence of GAIA improvements.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Iterable

HOP_ID = re.compile(r"^([34])hop[0-9]+__")
SEED = "musique-train-multihop-20261114"
SOURCE = "bdsaglam/musique@667064b514d22deecae00a3dccbd92bccf88869e"


def rows(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _normal(question: object) -> str:
    return " ".join(str(question or "").casefold().split())


def _hop(row: dict, *, excluded_ids: set[str], excluded_questions: set[str],
         excluded_decomp: set[str]) -> int | None:
    ident = str(row.get("id") or "")
    match = HOP_ID.match(ident)
    question = _normal(row.get("question"))
    answer = str(row.get("answer") or "").strip()
    decomposition = row.get("question_decomposition")
    passages = row.get("paragraphs")
    if (not match or row.get("answerable", True) is not True
        or ident in excluded_ids or question in excluded_questions
        or not question or not answer or len(answer) > 120
        or _normal(answer) in question
        or not isinstance(decomposition, list)
        or len(decomposition) != int(match[1])
        or not isinstance(passages, list) or len(passages) != 20):
        return None
    subids = [str(x["id"]) for x in decomposition
              if isinstance(x, dict) and x.get("id") is not None]
    if len(subids) != len(decomposition) or any(x in excluded_decomp for x in subids):
        return None
    indexes, supporting = set(), 0
    for part in passages:
        if not isinstance(part, dict) or not isinstance(part.get("idx"), int):
            return None
        idx = part["idx"]
        text = part.get("paragraph_text")
        if (idx in indexes or idx < 0 or idx > 99 or not isinstance(text, str)
            or not 20 <= len(text) <= 9000):
            return None
        indexes.add(idx)
        supporting += part.get("is_supporting") is True
    return int(match[1]) if supporting >= int(match[1]) else None


def select_training_tasks(train: Iterable[dict], dev: Iterable[dict], *,
                          limit: int = 8) -> list[dict]:
    """Balanced 3/4-hop source samples, never DEV/test and never solved traces."""
    if limit < 2 or limit > 32 or limit % 2:
        raise ValueError("limit must be an even number in 2..32")
    dev_rows = list(dev)
    excluded_ids = {str(r.get("id")) for r in dev_rows}
    excluded_questions = {_normal(r.get("question")) for r in dev_rows}
    excluded_decomp = {
        str(part["id"])
        for r in dev_rows for part in (r.get("question_decomposition") or [])
        if isinstance(part, dict) and part.get("id") is not None
    }
    best: dict[int, list[tuple[str, str, dict]]] = {3: [], 4: []}
    seen = set()
    per_hop = limit // 2
    for row in train:
        h = _hop(row, excluded_ids=excluded_ids,
                 excluded_questions=excluded_questions, excluded_decomp=excluded_decomp)
        if h is None:
            continue
        rid = str(row["id"])
        qnorm = _normal(row["question"])
        if rid in seen:
            raise ValueError("duplicate source id")
        seen.add(rid)
        score = sha256(f"{SEED}|{rid}".encode()).hexdigest()
        slot = best[h]
        slot.append((score, qnorm, row))
        slot.sort(key=lambda r: (r[0], r[1]))
        del slot[per_hop:]
    if any(len(best[h]) != per_hop for h in (3, 4)):
        raise ValueError(f"insufficient disjoint multi-hop TRAIN: 3hop={len(best[3])} 4hop={len(best[4])}")

    tasks = []
    for hop in (3, 4):
        for _score, _qnorm, row in best[hop]:
            paragraphs = sorted(row["paragraphs"], key=lambda p: p["idx"])
            index = "\n".join(
                f"passages/p{part['idx']}.txt | {part.get('title', '')}"
                for part in paragraphs
            )
            files = {"passages/index.txt": index}
            for part in paragraphs:
                files[f"passages/p{part['idx']}.txt"] = str(part["paragraph_text"])
            tasks.append({
                "id": "musique-train-" + str(row["id"]),
                "source": "musique-train",
                "license": "CC-BY-4.0",
                "generator": "official-multihop-reading-candidate",
                "generator_version": SOURCE,
                "oracle_type": "musique-compositional-human-annotated",
                "oracle_version": "MuSiQue-v1.0",
                "question": (
                    "Answer using ONLY workspace passage files. Start with "
                    "passages/index.txt, read relevant passage files using the read tool, "
                    "follow multiple evidence links, and give one concise final answer. "
                    "Question: " + str(row["question"])
                ),
                "expected_answer": str(row["answer"]),
                "required_tools": ["read"],
                "hop_count": hop,
                "source_id_hash": sha256(str(row["id"]).encode()).hexdigest(),
                "supporting_passages": sum(p.get("is_supporting") is True for p in paragraphs),
                "files": files,
            })
    assert len({t["question"] for t in tasks}) == len(tasks)
    return tasks


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--dev", type=Path, required=True)
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.report.exists() or a.output.resolve() == a.report.resolve():
        raise FileExistsError("refuse overwrite")
    candidates = select_training_tasks(rows(a.train), rows(a.dev), limit=a.limit)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True)+"\n"
                                for row in candidates), encoding="utf-8")
    report = {
        "source": SOURCE, "selected": len(candidates),
        "three_hop": sum(r["hop_count"] == 3 for r in candidates),
        "four_hop": sum(r["hop_count"] == 4 for r in candidates),
        "documents_per_task": 20,
        "train_sha256": sha256(a.train.read_bytes()).hexdigest(),
        "dev_sha256": sha256(a.dev.read_bytes()).hexdigest(),
        "observed_agent_successes": 0, "gaia_rows": 0, "training_ran": False,
        "boundary": "Source-verified candidate tasks only; do not train without actual observed multi-read success and GAIA protected-set checks",
    }
    a.report.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print("MUSIQUE_TRAIN_MULTIHOP_SOURCE_READY", json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
