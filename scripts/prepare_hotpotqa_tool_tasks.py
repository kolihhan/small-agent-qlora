"""Convert real HotpotQA multi-hop questions into offline local read-tool tasks.

Source: https://huggingface.co/datasets/hotpotqa/hotpot_qa (CC BY-SA 4.0).
Train split is for potential SFT; validation split is ONLY for independent eval.
No answers or supporting-fact labels enter the model-facing prompt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
import hashlib
import tempfile

from gaia_small_agent.training.protection import question_hash

DATASET = "hotpotqa/hotpot_qa"
DATASET_LICENSE = "CC-BY-SA-4.0"
# Content-addressed verified mirror of the canonical HotpotQA Parquet conversion.
# Dataset Viewer can return 5xx; never silently switch to arbitrary data.
_PARQUET_FALLBACK = {
    "train": (
        "https://huggingface.co/datasets/huang-haonan/hotpot_qa/resolve/main/distractor/train-00000-of-00002.parquet",
        "76d3bb3048a7cc73c1958107c0c5872a00d7e7d00c105b81e92f6769e7822e68",
    ),
    "validation": (
        "https://huggingface.co/datasets/huang-haonan/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet",
        "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6",
    ),
}
INSTRUCTION = (
    "Find the answer using only the local article files. Read index.txt first, "
    "then read the articles you need. Return only the short answer. Question: "
)


def _fetch_parquet_fallback(*, split: str, offset: int, count: int) -> list[dict]:
    """Read a bounded page from a checksum-pinned HotpotQA Parquet mirror."""
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("Data viewer failed; install pyarrow for pinned Parquet fallback") from exc
    url, expected_sha256 = _PARQUET_FALLBACK[split]
    with tempfile.TemporaryDirectory(prefix="hotpot-parquet-") as temp:
        target = Path(temp) / "source.parquet"
        digest = hashlib.sha256()
        length = 0
        with urlopen(url, timeout=120) as response, target.open("wb") as output:
            while block := response.read(1024 * 1024):
                length += len(block)
                if length > 200 * 1024 * 1024:
                    raise ValueError("unreasonably large Parquet download")
                output.write(block)
                digest.update(block)
        if digest.hexdigest() != expected_sha256:
            raise ValueError("pinned HotpotQA Parquet checksum mismatch")
        table = parquet.ParquetFile(target)
        results: list[dict] = []
        pos = 0
        for batch in table.iter_batches(batch_size=64):
            for row in batch.to_pylist():
                if pos >= offset and len(results) < count:
                    results.append(row)
                pos += 1
                if len(results) == count:
                    return results
    raise ValueError("pinned Parquet file has insufficient rows")


def fetch_rows(*, split: str, offset: int, count: int) -> list[dict]:
    if split not in {"train", "validation"} or offset < 0 or not 1 <= count <= 40:
        raise ValueError("split/offset/count invalid")
    query = urlencode({
        "dataset": DATASET, "config": "distractor", "split": split,
        "offset": offset, "length": count,
    })
    try:
        with urlopen("https://datasets-server.huggingface.co/rows?" + query, timeout=35) as response:
            payload = json.load(response)
        entries = payload.get("rows")
        if not isinstance(entries, list) or len(entries) != count:
            raise ValueError("incomplete HotpotQA dataset response")
        return [x["row"] for x in entries]
    except (HTTPError, URLError, TimeoutError) as error:
        # A public dataset mirror is acceptable only when the artifact matches
        # independently pinned SHA256 bytes. Never mask 400/401/403.
        if isinstance(error, HTTPError) and error.code < 500:
            raise
        return _fetch_parquet_fallback(split=split, offset=offset, count=count)


def convert_row(row: dict, *, split: str, max_article_chars: int = 1800) -> dict:
    if split not in {"train", "validation"} or max_article_chars < 300:
        raise ValueError("invalid conversion settings")
    context = row["context"]
    titles = context["title"]
    sentences = context["sentences"]
    fact_titles = row["supporting_facts"]["title"]
    if (not isinstance(titles, list) or not isinstance(sentences, list)
        or len(titles) != len(sentences) or not 2 <= len(titles) <= 20
        or len(set(titles)) != len(titles)
        or not set(fact_titles).issubset(titles) or len(set(fact_titles)) < 2):
        raise ValueError("not a well-formed two-article multi-hop record")
    question, answer = row["question"], row["answer"]
    if not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or not answer.strip():
        raise ValueError("missing task or answer")
    files: dict[str, str] = {}
    index_lines = ["Articles available in the workspace:"]
    for index, (title, parts) in enumerate(zip(titles, sentences)):
        if (not isinstance(title, str) or not isinstance(parts, list)
            or not all(isinstance(sentence, str) for sentence in parts)):
            raise ValueError("invalid article")
        body = "Article: " + title + "\n" + "\n".join(parts)
        if len(body) > max_article_chars:
            raise ValueError("article too long; do not truncate evidence")
        filename = f"article_{index:02d}.txt"
        files[filename] = body
        index_lines.append(f"{filename}: {title}")
    files["index.txt"] = "\n".join(index_lines)
    task_id = str(row["id"])
    if not task_id:
        raise ValueError("missing stable row id")
    return {
        "id": "hotpot-" + split + "-" + task_id,
        "source": "hotpotqa-distractor-" + split,
        "license": DATASET_LICENSE,
        "generator": "hotpotqa-offline-read-agent",
        "generator_version": "1",
        "oracle_type": "human-annotated-exact-answer",
        "oracle_version": "hotpotqa-v1",
        "question": INSTRUCTION + question,
        "expected_answer": answer,
        "required_tools": ["read"],
        "files": files,
        "task_level": row.get("level", "unknown"),
        "task_type": row.get("type", "unknown"),
        "dataset_attribution": "Yang et al., HotpotQA, EMNLP 2018 (CC BY-SA 4.0)",
    }


def convert_page(rows: list[dict], *, split: str, limit: int) -> tuple[list[dict], dict]:
    if limit < 1:
        raise ValueError("limit must be positive")
    selected: list[dict] = []
    rejected: dict[str, int] = {}
    for row in rows:
        if len(selected) >= limit:
            break
        try:
            selected.append(convert_row(row, split=split))
        except (KeyError, TypeError, ValueError) as error:
            reason = str(error)
            rejected[reason] = rejected.get(reason, 0) + 1
    if not selected:
        raise ValueError("no acceptable independent multi-hop tasks")
    return selected, {
        "dataset": DATASET, "source_split": split,
        "license": DATASET_LICENSE, "input_rows": len(rows),
        "accepted": len(selected), "rejected": rejected,
        "gold_hidden_from_prompt": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--holdout-hashes", type=Path)
    args = parser.parse_args()
    if args.split == "train" and args.holdout_hashes:
        raise ValueError("train must not create evaluation hash file")
    if args.split == "validation" and not args.holdout_hashes:
        raise ValueError("validation must create protected evaluation hashes")
    for path in (args.output, args.report):
        if path.exists():
            raise FileExistsError("refusing to overwrite frozen task files")
    if args.holdout_hashes and args.holdout_hashes.exists():
        raise FileExistsError("refusing to overwrite protected evaluation hashes")
    rows = fetch_rows(split=args.split, offset=args.offset, count=args.count)
    tasks, report = convert_page(rows, split=args.split, limit=args.limit)
    for path in (args.output, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in tasks),
        encoding="utf-8",
    )
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.holdout_hashes:
        hashes = sorted({question_hash(row["question"]) for row in tasks})
        args.holdout_hashes.parent.mkdir(parents=True, exist_ok=True)
        args.holdout_hashes.write_text(
            json.dumps({"algorithm": "sha256-normalized-question-v1", "hashes": hashes}, indent=2)+"\n",
            encoding="utf-8",
        )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
