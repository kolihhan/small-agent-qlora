import json
import pytest
from gaia_small_agent.training.qlora import load_verified_rows


def _verified_row(*, source="synthetic-recovery", question="x", answer="y"):
    return {
        "task_id": "unit-1",
        "verified": True,
        "source": source,
        "expected_answer": answer,
        "required_tools": [],
        "provenance": {
            "license": "CC0-1.0",
            "generator": "unit-test",
            "generator_version": "1",
            "oracle_type": "exact",
            "oracle_version": "1",
        },
        "verification": {
            "completed": True,
            "final_answer_correct": True,
            "zero_tool_errors": True,
            "zero_duplicate_blocks": True,
            "required_tools_satisfied": True,
            "provenance_complete": True,
        },
        "tools": [],
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
    }


def test_training_rejects_gaia_trajectory(tmp_path):
    p = tmp_path / "rows.jsonl"
    row = _verified_row(source="GAIA-validation")
    p.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="GAIA trajectories are forbidden"):
        load_verified_rows(p)


def test_training_accepts_verified_non_gaia_trajectory(tmp_path):
    p = tmp_path / "rows.jsonl"
    row = _verified_row()
    p.write_text(json.dumps(row) + "\n")
    assert load_verified_rows(p) == [row]


def test_training_rejects_gaia_marker_anywhere_in_source(tmp_path):
    p = tmp_path / "rows.jsonl"
    row = _verified_row(source="synthetic-gaia-derived")
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="GAIA trajectories are forbidden"):
        load_verified_rows(p)


def test_training_rejects_verified_flag_without_complete_provenance_and_policy_checks(tmp_path):
    p = tmp_path / "rows.jsonl"
    row = _verified_row()
    row["provenance"] = {"license": "CC0-1.0"}
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        load_verified_rows(p)


def test_training_rejects_dirty_policy_trace_even_when_verified_flag_is_true(tmp_path):
    p = tmp_path / "rows.jsonl"
    row = _verified_row()
    row["verification"]["zero_tool_errors"] = False
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="policy verification"):
        load_verified_rows(p)


def test_turn_rendering_fails_closed_if_chat_template_cannot_isolate_native_completion():
    from gaia_small_agent.training.qlora import _render_turn_examples

    class BadTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            if kwargs.get("add_generation_prompt"):
                return "PROMPT-WITH-ASSISTANT-OPENER"
            return "DIFFERENT-NATIVE-RENDER"

    row = _verified_row()
    with pytest.raises(ValueError, match="native format"):
        _render_turn_examples(BadTokenizer(), [row])


def test_oracle_tool_and_no_tool_rows_render_as_next_action_examples(tmp_path):
    from gaia_small_agent.training.oracle_policy import generate_oracle_policy_dataset
    from gaia_small_agent.training.qlora import _render_turn_examples

    paths = generate_oracle_policy_dataset(tmp_path / "oracle", seed="render")
    rows = [json.loads(line) for line in paths["dev"].read_text(encoding="utf-8").splitlines()]
    tool_row = next(row for row in rows if row["capability"] == "tool_selection")
    no_tool_row = next(row for row in rows if row["capability"] == "no_tool_stop")

    class NativeShapeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            def dump(value):
                return json.dumps(value, sort_keys=True, separators=(",", ":"))

            if kwargs.get("add_generation_prompt"):
                return dump(messages) + "<assistant>"
            if messages and messages[-1].get("role") == "assistant":
                return dump(messages[:-1]) + "<assistant>" + dump(messages[-1])
            return dump(messages)

    tool_examples = _render_turn_examples(NativeShapeTokenizer(), [tool_row])
    no_tool_examples = _render_turn_examples(NativeShapeTokenizer(), [no_tool_row])

    assert len(tool_examples) == 2
    assert any("tool_calls" in example["completion"] for example in tool_examples)
    assert any(tool_row["expected_answer"] in example["completion"] for example in tool_examples)
    assert len(no_tool_examples) == 1
    assert no_tool_row["expected_answer"] in no_tool_examples[0]["completion"]


def test_cpu_safe_loss_uses_native_model_loss_without_fused_lm_head():
    from gaia_small_agent.training.qlora import _cpu_safe_compute_loss

    calls = []

    class Output:
        loss = 1.25

    class Model:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return Output()

    loss = _cpu_safe_compute_loss(Model(), {"input_ids": [1], "labels": [1]})

    assert loss == 1.25
    assert calls == [{"input_ids": [1], "labels": [1]}]


def test_cpu_safe_loss_can_return_outputs():
    from gaia_small_agent.training.qlora import _cpu_safe_compute_loss

    class Output:
        loss = 2.5

    output = Output()

    class Model:
        def __call__(self, **kwargs):
            return output

    loss, returned = _cpu_safe_compute_loss(Model(), {"input_ids": [1], "labels": [1]}, return_outputs=True)

    assert loss == 2.5
    assert returned is output
