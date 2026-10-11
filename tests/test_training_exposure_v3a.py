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


def _turn_examples(tmp_path):
    from gaia_small_agent.training.oracle_policy_v2 import generate_oracle_policy_v2_dataset
    from gaia_small_agent.training.qlora import _render_turn_examples

    paths = generate_oracle_policy_v2_dataset(tmp_path / "oracle", seed="small-agent-policy-v2")
    rows = [json.loads(line) for line in paths["train"].read_text(encoding="utf-8").splitlines()]
    return _render_turn_examples(NativeShapeTokenizer(), rows)


def test_turn_examples_carry_trace_metadata_without_leaking_it_to_tokens(tmp_path):
    from gaia_small_agent.training.qlora import _tokenize_completion_examples

    examples = _turn_examples(tmp_path)

    assert len(examples) == 3600
    first = examples[0]
    assert set(first["metadata"]) == {
        "example_id",
        "task_id",
        "capability",
        "action_type",
        "assistant_message_index",
    }
    assert first["metadata"]["example_id"] == (
        f"{first['metadata']['task_id']}:{first['metadata']['assistant_message_index']}"
    )
    assert first["metadata"]["action_type"] in {"tool_call", "final"}

    tokenized = _tokenize_completion_examples(NativeShapeTokenizer(), examples[:3], max_length=512)
    assert set(tokenized[0]) == {"input_ids", "attention_mask", "labels"}


def test_v3a_exposure_selection_is_balanced_unique_and_deterministic(tmp_path):
    from gaia_small_agent.training.qlora import (
        _build_training_exposure_report,
        _select_balanced_exposure_examples,
    )

    examples = _turn_examples(tmp_path)
    selected = _select_balanced_exposure_examples(examples, count=64, seed="qlora-v3a-exposure")
    repeated = _select_balanced_exposure_examples(examples, count=64, seed="qlora-v3a-exposure")

    assert len(selected) == 64
    assert len({item["metadata"]["example_id"] for item in selected}) == 64
    assert len({item["metadata"]["task_id"] for item in selected}) == 64
    assert [item["metadata"]["example_id"] for item in selected] == [
        item["metadata"]["example_id"] for item in repeated
    ]

    capability_counts = Counter(item["metadata"]["capability"] for item in selected)
    assert capability_counts == {
        "tool_selection": 8,
        "argument_grounding": 8,
        "multi_step": 8,
        "evidence_to_final": 8,
        "no_tool_stop": 8,
        "failure_recovery": 8,
        "strategy_switch": 8,
        "duplicate_avoidance": 8,
    }

    by_capability = defaultdict(Counter)
    for item in selected:
        meta = item["metadata"]
        by_capability[meta["capability"]][meta["action_type"]] += 1

    assert by_capability["no_tool_stop"] == {"final": 8}
    for capability in capability_counts:
        if capability != "no_tool_stop":
            assert by_capability[capability] == {"tool_call": 5, "final": 3}

    action_counts = Counter(item["metadata"]["action_type"] for item in selected)
    assert action_counts == {"tool_call": 35, "final": 29}

    report = _build_training_exposure_report(
        selected,
        seed="qlora-v3a-exposure",
        source_example_count=len(examples),
    )
    repeated_report = _build_training_exposure_report(
        repeated,
        seed="qlora-v3a-exposure",
        source_example_count=len(examples),
    )
    assert report["selected_example_count"] == 64
    assert report["source_example_count"] == 3600
    assert report["selection_sha256"] == repeated_report["selection_sha256"]
    assert len(report["selection_sha256"]) == 64
    assert report["action_type_counts"] == {"final": 29, "tool_call": 35}
    assert report["capability_counts"] == dict(sorted(capability_counts.items()))
