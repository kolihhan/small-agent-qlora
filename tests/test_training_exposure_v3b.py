from __future__ import annotations

from collections import Counter, defaultdict
import json


class NativeShapeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        def dump(value):
            return json.dumps(value, sort_keys=True, separators=(",", ":"))

        if kwargs.get("add_generation_prompt"):
            return dump(messages) + "<assistant>"
        if messages and messages[-1].get("role") == "assistant":
            return dump(messages[:-1]) + "<assistant>" + dump(messages[-1])
        return dump(messages)

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [(ord(char) % 251) + 1 for char in text]}


def test_v3b_exposure_is_diagnostic_weighted_unique_and_deterministic(tmp_path):
    from gaia_small_agent.training.oracle_policy_v3 import generate_oracle_policy_v3_dataset
    from gaia_small_agent.training.qlora import (
        _build_training_exposure_report,
        _render_turn_examples,
        _select_diagnostic_exposure_examples,
    )

    paths = generate_oracle_policy_v3_dataset(tmp_path / "oracle", seed="small-agent-policy-v3")
    rows = [json.loads(line) for line in paths["train"].read_text(encoding="utf-8").splitlines()]
    examples = _render_turn_examples(NativeShapeTokenizer(), rows)

    selected = _select_diagnostic_exposure_examples(examples, count=64, seed="qlora-v3b-exposure")
    repeated = _select_diagnostic_exposure_examples(examples, count=64, seed="qlora-v3b-exposure")

    assert len(selected) == 64
    assert len({item["metadata"]["example_id"] for item in selected}) == 64
    assert len({item["metadata"]["task_id"] for item in selected}) == 64
    assert [item["metadata"]["example_id"] for item in selected] == [
        item["metadata"]["example_id"] for item in repeated
    ]

    by_capability = defaultdict(Counter)
    for item in selected:
        metadata = item["metadata"]
        by_capability[metadata["capability"]][metadata["action_type"]] += 1

    assert by_capability == {
        "evidence_to_final": Counter({"final": 12, "tool_call": 8}),
        "multi_step": Counter({"tool_call": 8, "final": 4}),
        "failure_recovery": Counter({"tool_call": 5, "final": 3}),
        "strategy_switch": Counter({"tool_call": 4, "final": 2}),
        "argument_grounding": Counter({"tool_call": 4, "final": 2}),
        "tool_selection": Counter({"tool_call": 4, "final": 2}),
        "duplicate_avoidance": Counter({"tool_call": 2, "final": 2}),
        "no_tool_stop": Counter({"final": 2}),
    }

    report = _build_training_exposure_report(
        selected,
        seed="qlora-v3b-exposure",
        source_example_count=len(examples),
    )
    assert report["selected_example_count"] == 64
    assert report["action_type_counts"] == {"final": 27, "tool_call": 37}
    assert report["capability_counts"] == {
        "argument_grounding": 6,
        "duplicate_avoidance": 4,
        "evidence_to_final": 20,
        "failure_recovery": 8,
        "multi_step": 12,
        "no_tool_stop": 2,
        "strategy_switch": 6,
        "tool_selection": 6,
    }
