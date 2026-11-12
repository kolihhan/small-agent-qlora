"""Fail-closed rejection sampling of *observed* tool trajectories for next-action SFT.

Input is JSONL from actual AgentRuntime collections, not hand-written teacher SQL
or model-generated narratives. This verifies message/action consistency, NOT the
truth of external tool outputs; an independent trusted task oracle is required.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from gaia_small_agent.benchmark.scoring import gaia_score
from gaia_small_agent.training.protection import load_protected_question_hashes, question_hash

_REQUIRED_PROVENANCE = ("license", "generator", "generator_version", "oracle_type", "oracle_version")
_REQUIRED_VERIFICATION = (
    "completed", "final_answer_correct", "zero_duplicate_blocks",
    "required_tools_satisfied", "provenance_complete",
)
_SAMPLE_TAG = "::sample-"


def _check_row(row: dict, protected_hashes: set[str], holdout_hashes: set[str], min_actions: int) -> tuple[list[int], dict]:
    """Return trainable assistant indices, or raise on an unsafe candidate."""
    if row.get("verified") is not True:
        raise ValueError("not_verified")
    provenance = row.get("provenance")
    if not isinstance(provenance, dict) or any(not provenance.get(k) for k in _REQUIRED_PROVENANCE):
        raise ValueError("missing_provenance")
    source = str(row.get("source") or "")
    if not source or "gaia" in source.casefold():
        raise ValueError("forbidden_source")
    multi_hop_train_sources = {"hotpotqa-distractor-train", "2wikimultihopqa-train"}
    if ((source.startswith("hotpotqa-distractor-") or
         source.startswith("2wikimultihopqa-")) and
        source not in multi_hop_train_sources):
        # Explicit split pin: caller-supplied holdout hashes may be incomplete.
        raise ValueError("forbidden_multihop_holdout_split")
    grounded_multihop = source in multi_hop_train_sources
    verification = row.get("verification")
    if not isinstance(verification, dict) or any(verification.get(k) is not True for k in _REQUIRED_VERIFICATION):
        raise ValueError("unverified_process")
    if verification.get("zero_tool_errors") is not True:
        # The stock collector marks failed tool calls ineligible. To recover from
        # failed calls later, a separately attested process verifier is required.
        raise ValueError("tool_error_not_attested")
    metrics = row.get("metrics")
    if not isinstance(metrics, dict) or int(metrics.get("duplicate_calls_blocked", -1)) != 0:
        raise ValueError("duplicate_tool_calls")
    if int(metrics.get("tool_errors", -1)) != 0:
        raise ValueError("tool_errors")
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 4:
        raise ValueError("missing_messages")
    users = [m for m in messages if isinstance(m, dict) and m.get("role") == "user"]
    if len(users) != 1 or not isinstance(users[0].get("content"), str):
        raise ValueError("ambiguous_question")
    question = users[0]["content"]
    h = question_hash(question)
    if h in protected_hashes or h in holdout_hashes:
        raise ValueError("protected_or_holdout_question")
    if not isinstance(messages[-1], dict) or messages[-1].get("role") != "assistant" or messages[-1].get("tool_calls"):
        raise ValueError("missing_final")
    final = messages[-1].get("content")
    expected = row.get("expected_answer")
    if not isinstance(final, str) or not final.strip() or not isinstance(expected, str) or not gaia_score(final, expected):
        raise ValueError("wrong_final")

    available_tools = set()
    for tool in row.get("tools") or []:
        if isinstance(tool, dict):
            fun = tool.get("function") or {}
            if isinstance(fun, dict) and isinstance(fun.get("name"), str):
                available_tools.add(fun["name"])
    used_ids: set[str] = set()
    seen_calls: set[str] = set()
    # HotpotQA acceptance is NOT only final-answer matching: require the model
    # to have actually read an index and >=2 distinct local articles.
    hotpot_read_sources: set[str] = set()
    hotpot_article_observations: list[str] = []
    trainable: list[int] = []
    tool_calls = 0
    assistant_tool_turns = 0
    position = 0
    tool_observation_chars = 0
    while position < len(messages):
        msg = messages[position]
        if not isinstance(msg, dict):
            raise ValueError("bad_message")
        if msg.get("role") == "tool":
            raise ValueError("orphan_tool_result")
        if msg.get("role") != "assistant":
            position += 1
            continue
        calls = msg.get("tool_calls") or []
        if not calls:
            if position != len(messages) - 1:
                raise ValueError("early_final")
            trainable.append(position)
            position += 1
            continue
        if not isinstance(calls, list):
            raise ValueError("bad_tool_call")
        for call in calls:
            if not isinstance(call, dict):
                raise ValueError("bad_tool_call")
            fun = call.get("function") or {}
            call_id = call.get("id")
            if not isinstance(fun, dict) or not isinstance(call_id, str) or not call_id:
                raise ValueError("bad_tool_call")
            name = fun.get("name")
            arguments = fun.get("arguments")
            if name not in available_tools or not isinstance(arguments, dict):
                raise ValueError("unexpected_tool")
            signature = json.dumps((name, arguments), sort_keys=True, default=str)
            if signature in seen_calls or call_id in used_ids:
                raise ValueError("duplicate_tool_call")
            seen_calls.add(signature)
            used_ids.add(call_id)
        results = messages[position + 1: position + 1 + len(calls)]
        if len(results) != len(calls) or any(not isinstance(m, dict) or m.get("role") != "tool" for m in results):
            raise ValueError("unpaired_tool_call")
        expected_ids = {str(call["id"]) for call in calls}
        observed_ids = [str(m.get("tool_call_id")) for m in results]
        if set(observed_ids) != expected_ids or len(set(observed_ids)) != len(observed_ids):
            raise ValueError("mismatched_tool_result")
        observed_by_id = {str(m["tool_call_id"]): m for m in results}
        for call in calls:
            result = observed_by_id[call["id"]]
            if result.get("name") != call["function"]["name"]:
                raise ValueError("mismatched_tool_result")
            contents = result.get("content")
            if not isinstance(contents, str):
                raise ValueError("invalid_tool_observation")
            if contents.startswith(("ToolError[", "ERROR[")):
                raise ValueError("failed_tool_action")
            tool_observation_chars += len(contents)
            function = call["function"]
            if function["name"] == "read":
                filename = function["arguments"].get("source")
                if isinstance(filename, str):
                    hotpot_read_sources.add(filename)
                    if filename.startswith("article_") and filename.endswith(".txt"):
                        hotpot_article_observations.append(contents)
        tool_calls += len(calls)
        assistant_tool_turns += 1
        trainable.append(position)
        position += len(calls) + 1

    if tool_calls < min_actions:
        raise ValueError("too_shallow")
    if grounded_multihop:
        # Three generic read calls are not proof of two-hop grounding.
        articles = {filename for filename in hotpot_read_sources
                    if filename.startswith("article_") and filename.endswith(".txt")}
        if "index.txt" not in hotpot_read_sources or len(articles) < 2:
            raise ValueError("hotpot_missing_index_or_two_articles")
        # For the initial extractive curriculum, reject correct-by-chance finals
        # when the answer does not appear in either observed article. Yes/no and
        # nonextractive answers need a separate evidence entailment verifier.
        answer = expected.strip().casefold()
        if answer in {"yes", "no"} or not any(
            answer in observation.casefold() for observation in hotpot_article_observations
        ):
            raise ValueError("hotpot_answer_not_observed_or_nonextractive")
    if tool_calls != int(metrics.get("tool_calls", -1)):
        raise ValueError("tool_metric_mismatch")
    if tool_observation_chars == 0:
        raise ValueError("missing_tool_evidence")
    if len(trainable) != assistant_tool_turns + 1:
        # One model-completion target per assistant turn, even when it executes
        # multiple valid tools in parallel. This preserves native message shape.
        raise ValueError("assistant_action_count_mismatch")
    return trainable, {
        "question_hash": h,
        "tool_calls": tool_calls,
        "final_assistant_index": trainable[-1],
        "train_actions": len(trainable),
        "observation_chars": tool_observation_chars,
        "hotpot_distinct_article_reads": len({
            x for x in hotpot_read_sources if x.startswith("article_") and x.endswith(".txt")
        }) if grounded_multihop else None,
    }


def curate_teacher_rows(
    rows: list[dict], *, protected_hashes: set[str], holdout_hashes: set[str],
    min_actions: int = 3, min_long_trajectories: int = 1,
) -> tuple[list[dict], dict]:
    if not protected_hashes:
        raise ValueError("GAIA-protected question hash set must be nonempty")
    if min_actions < 1 or min_long_trajectories < 1:
        raise ValueError("positive curriculum gates required")
    grouped: dict[str, list[tuple[dict, list[int], dict]]] = defaultdict(list)
    rejected: Counter[str] = Counter()
    for row in rows:
        try:
            indices, stats = _check_row(row, protected_hashes, holdout_hashes, min_actions)
            task_id = str(row.get("task_id") or "")
            if not task_id:
                raise ValueError("missing_task_id")
            parent = task_id.split(_SAMPLE_TAG, 1)[0]
            if not parent:
                raise ValueError("invalid_task_id")
            grouped[parent].append((row, indices, stats))
        except (ValueError, TypeError, KeyError) as err:
            rejected[str(err)] += 1
    selected: list[dict] = []
    unique_questions: set[str] = set()
    for parent in sorted(grouped):
        candidates = grouped[parent]
        # Rejection sampling chooses the shortest clean successful trajectory
        # among trials of the SAME task (not the easiest tasks globally).
        candidates.sort(key=lambda pair: (
            pair[2]["tool_calls"],
            pair[2]["final_assistant_index"],
            sha256(str(pair[0]["task_id"]).encode()).hexdigest(),
        ))
        row, indices, stats = candidates[0]
        if stats["question_hash"] in unique_questions:
            rejected["duplicate_question_across_tasks"] += len(candidates)
            continue
        unique_questions.add(stats["question_hash"])
        curated = dict(row)
        curated["task_id"] = parent
        curated["training_action_indices"] = indices
        curated["selection_provenance"] = {
            "method": "best-of-n-clean-observed-rollouts",
            "sample_id": str(row["task_id"]),
            "candidate_count": len(candidates),
            "tool_calls": stats["tool_calls"],
            "final_assistant_index": stats["final_assistant_index"],
        }
        selected.append(curated)
    report = {
        "schema": "teacher-trajectory-rejection-sampling/v1",
        "input_rollouts": len(rows),
        "accepted_tasks": len(selected),
        "rejected_reasons": dict(sorted(rejected.items())),
        "final_depth_histogram": dict(sorted(Counter(
            str(row["selection_provenance"]["final_assistant_index"]) for row in selected
        ).items())),
        "trainable_actions": sum(len(row["training_action_indices"]) for row in selected),
        "source_policy": "non-GAIA, exact known answer, clean observed tool actions",
        "warning": "Tool contents are not independently attested by this structural verifier.",
    }
    if len(selected) < min_long_trajectories:
        raise ValueError(f"insufficient verified long trajectories: {len(selected)} < {min_long_trajectories}; report={report}")
    return selected, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSONL from observed tool rollouts")
    parser.add_argument("--protected", type=Path, required=True, help="GAIA question hash set; nonempty")
    parser.add_argument("--holdout-hashes", type=Path, required=True, help="Independent evaluation question hash set")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-actions", type=int, default=3)
    parser.add_argument("--min-long-trajectories", type=int, default=1)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected, report = curate_teacher_rows(
        rows, protected_hashes=load_protected_question_hashes(args.protected),
        holdout_hashes=load_protected_question_hashes(args.holdout_hashes),
        min_actions=args.min_actions, min_long_trajectories=args.min_long_trajectories,
    )
    for path in (args.output, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refuse to replace existing curated training data or report")
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
