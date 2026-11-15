"""Validate whether IIRC TRAIN offers observable evidence-to-read/stop labels."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from gaia_small_agent.agent.loop import MAX_TOOL_OBSERVATION_CHARS
from gaia_small_agent.tools import ReadTool
from gaia_small_agent.training.trajectories import _materialize_files
from scripts.eval_iirc_links import (
    ANCHOR, all_visible, key, load_articles, load_split, published_support, rank, workspace,
)


PAGE_BUDGET = 128


def observed_pages(tool: ReadTool, root: Path, path: str, spans: list[str],
                   max_pages: int) -> tuple[str, int]:
    body = ""
    offset = 0
    for page in range(1, max_pages + 1):
        result = tool.run({"source": path, "offset_chars": offset}, root)
        if not result.ok:
            raise RuntimeError(f"tool rejected {path}: {result.error_code}")
        obs, metadata = result.bounded_observation(MAX_TOOL_OBSERVATION_CHARS)
        visible = obs.partition("\n")[2][:metadata["returned_chars"]]
        if metadata["offset_chars"] != offset:
            raise ValueError("ReadTool offsets disagree with the requested continuation")
        body += visible
        next_offset = metadata["next_offset_chars"]
        if all_visible(spans, body) or next_offset is None:
            return body, page
        if next_offset != offset + len(visible) or next_offset <= offset:
            raise ValueError("read continuation would repeat or skip visible text")
        offset = next_offset
    return body, max_pages


def select_next_targets(page: dict, row: dict, paths: dict, support: dict, main: str) -> dict | None:
    asked = {key(name) for name in row["question_links"]}
    linked = {name: path for name, path in paths.items() if path in support}
    if not linked.keys() <= asked:
        return None
    verified = {}
    for anchor in page["links"]:
        name = key(anchor["target"])
        path = linked.get(name)
        if not path:
            continue
        start, end = anchor["indices"]
        mention = key(page["text"][start:end])
        if mention and f"{mention} [{path}]" in key(main) and any(
            mention in excerpt for excerpt in support["main.txt"]
        ):
            verified[path] = name
    return verified if len(verified) == len(linked) else None


def audit(source: Path, articles_path: Path, cohort: dict, previous: Path,
          max_pages: int) -> dict:
    train = load_split(source, "train")
    dev = load_split(source, "dev")
    prior_workspaces = (json.loads(line)["eval_task"]["files"] for line in previous.open(encoding="utf-8"))
    blocked = {key(label) for files in prior_workspaces for line in files["passages/index.txt"].splitlines()
               if (label := line.partition(" | ")[2])}
    blocked.update(key(x["title"]) for x in dev)
    for case in cohort["cases"]:
        blocked.update(case["support_titles"])
        blocked.update(key(line.partition(" | ")[2]) for line in case["files"]["index.txt"].splitlines())
    excluded_qids = {row["qid"] for page in dev for row in page["questions"]}
    prioritized = sorted(
        (page for page in train if key(page["title"]) not in blocked and 6 <= len(page["links"]) <= 30 and any(
            row["answer"]["type"] == "span" and 2 <= len(set(row["question_links"])) <= 3
            for row in page["questions"]
        )),
        key=lambda page: rank("train:" + page["pid"]),
    )[:PAGE_BUDGET]
    needed = {key(link["target"]) for page in prioritized for link in page["links"]}
    articles = load_articles(articles_path, needed)
    counts = Counter()
    eligible = []
    read = ReadTool()
    for page in prioritized:
        files, paths, linked = workspace(page, articles)
        files["main.txt"] = linked
        for row in sorted(page["questions"], key=lambda row: rank(row["qid"])):
            counts["questions_considered"] += 1
            span = row["answer"].get("answer_spans", [])
            if row["qid"] in excluded_qids or row["answer"]["type"] != "span" or len(span) != 1:
                continue
            support = published_support(row, paths)
            if not support:
                continue
            titles = {name for name, path in paths.items() if path in support}
            if titles & blocked:
                counts["excluded_title_overlap"] += 1
                continue
            with TemporaryDirectory(prefix="iirc-train-audit-") as temporary:
                root = Path(temporary)
                _materialize_files({"files": files}, root)
                main, _ = observed_pages(read, root, "main.txt", support["main.txt"], 1)
                if not all_visible(support["main.txt"], main):
                    counts["main_support_not_visible"] += 1
                    continue
                targets = select_next_targets(page, row, paths, support, main)
                if not targets:
                    counts["missing_visible_annotated_anchor"] += 1
                    continue
                bodies = {path: observed_pages(read, root, path, support[path], max_pages) for path in targets}
                if not all(all_visible(support[path], body) for path, (body, _) in bodies.items()):
                    counts["linked_support_not_visible"] += 1
                    continue
            decisions = 2 + sum(pages for _, pages in bodies.values())
            if decisions > 8:
                counts["decision_budget_exceeded"] += 1
                continue
            answer = key(span[0]["text"])
            if not answer or not any(answer in excerpt for path in targets for excerpt in support[path]):
                counts["no_literal_in_linked_support"] += 1
                continue
            if any(answer in key(text) for text in (row["question"], ANCHOR.sub("", main))):
                counts["answer_literal_in_initial_state"] += 1
                continue
            eligible.append({
                "qid": row["qid"], "page": page["pid"], "page_title": key(page["title"]),
                "valid_read_paths": sorted(targets), "support_titles": sorted(titles),
                "verified_next_offset_actions": sum(pages - 1 for _, pages in bodies.values()),
                "read_and_final_decisions": decisions,
                "candidate_final_state_has_all_annotated_support": True,
            })
            break
    return {
        "protocol": "iirc-train-grounded-action-feasibility-v1",
        "train_pages_examined": len(prioritized),
        "train_parents_passing": len(eligible),
        "physically_verified_candidate_read_actions": sum(len(row["valid_read_paths"]) for row in eligible),
        "eligible_final_states": len(eligible),
        "max_observations_per_linked_page": max_pages,
        "physically_verified_continuation_actions": sum(row["verified_next_offset_actions"] for row in eligible),
        "eligible_seven_or_eight_decision_parents": sum(row["read_and_final_decisions"] >= 7 for row in eligible),
        "exclusions": dict(counts),
        "frozen_dev_cohort_sha": cohort["cohort_sha"],
        "labels_exported_for_training": 0,
        "cases": eligible,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("source", "articles", "cohort", "previous", "output"):
        p.add_argument("--" + field, required=True, type=Path)
    p.add_argument("--max-pages", type=int, choices=(1, 3), default=1)
    args = p.parse_args()
    summary = audit(
        args.source, args.articles, json.loads(args.cohort.read_text()), args.previous, args.max_pages,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
