"""Audit the exact 64-row v3B exposure selection for supervised failed first actions.

This reconstructs metadata from the deterministic oracle without loading a model.
It does not change the frozen v3B experiment or estimate GAIA accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from gaia_small_agent.training.oracle_policy_v3 import generate_oracle_policy_v3_dataset
from gaia_small_agent.training.qlora import _select_diagnostic_exposure_examples


def audit_v3b_selected_actions() -> dict:
    with TemporaryDirectory(prefix="v3b-action-audit-") as temp:
        paths = generate_oracle_policy_v3_dataset(Path(temp), seed="small-agent-policy-v3")
        rows = [
            json.loads(line)
            for line in paths["train"].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        example_by_id: dict[str, dict] = {}
        for row in rows:
            messages = row["messages"]
            for index, msg in enumerate(messages):
                if msg.get("role") != "assistant" or index == 0:
                    continue
                example_id = f'{row["task_id"]}:{index}'
                example_by_id[example_id] = {
                    "metadata": {
                        "example_id": example_id,
                        "task_id": row["task_id"],
                        "capability": row["capability"],
                        "action_type": "tool_call" if msg.get("tool_calls") else "final",
                        "assistant_message_index": index,
                    },
                    "next_tool_result": (
                        str(messages[index + 1].get("content") or "")
                        if msg.get("tool_calls") and index + 1 < len(messages)
                        else ""
                    ),
                }
        selected = _select_diagnostic_exposure_examples(
            list(example_by_id.values()), count=64, seed="qlora-v3b-exposure",
        )
    unsafe = []
    counts = Counter()
    for entry in selected:
        m = entry["metadata"]
        counts[m["capability"] + "/" + m["action_type"]] += 1
        # Oracle policy v3 has deliberate SEARCH_FIXTURE_MISS actions in
        # recovery/switch trajectories; those actions were not optimal first moves.
        if m["action_type"] == "tool_call" and "ERROR[SEARCH_FIXTURE_MISS]" in entry["next_tool_result"]:
            unsafe.append({
                "example_id": m["example_id"],
                "capability": m["capability"],
                "assistant_message_index": m["assistant_message_index"],
            })
    if len(selected) != 64 or len({x["metadata"]["task_id"] for x in selected}) != 64:
        raise RuntimeError("v3B selection changed or duplicated tasks")
    return {
        "schema": "v3b-frozen-supervision-audit/v1",
        "source": "1600 deterministic synthetic oracle train tasks; exact v3B metadata selection",
        "selected": len(selected),
        "unsafe_failed_first_actions": len(unsafe),
        "unsafe_by_capability": dict(sorted(Counter(x["capability"] for x in unsafe).items())),
        "selected_counts": dict(sorted(counts.items())),
        "unsafe_action_ids": [x["example_id"] for x in unsafe],
        "interpretation": "Potential erroneous supervised tool choices; not proof of generalization cause.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_v3b_selected_actions()
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
