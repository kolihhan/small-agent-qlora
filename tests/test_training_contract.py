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


def test_training_precision_uses_float32_without_mixed_precision_on_cpu():
    from gaia_small_agent.training.qlora import _training_precision

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def is_bf16_supported():
            return False

    class FakeTorch:
        cuda = FakeCuda()
        float32 = "float32"
        float16 = "float16"
        bfloat16 = "bfloat16"

    dtype, bf16, fp16 = _training_precision(FakeTorch())

    assert dtype == "float32"
    assert bf16 is False
    assert fp16 is False


def test_training_rejects_max_length_that_truncates_completion_tokens():
    from gaia_small_agent.training.qlora import _validate_example_token_budget

    class FakeTokenizer:
        def __call__(self, text, *, add_special_tokens=False):
            return {"input_ids": text.split()}

    examples = [{"prompt": "p p p p p", "completion": "c c"}]

    with pytest.raises(ValueError, match=r"max_length=7.*requires at least 8"):
        _validate_example_token_budget(FakeTokenizer(), examples, 7)


def test_training_accepts_max_length_that_preserves_prompt_completion_and_eos():
    from gaia_small_agent.training.qlora import _validate_example_token_budget

    class FakeTokenizer:
        def __call__(self, text, *, add_special_tokens=False):
            return {"input_ids": text.split()}

    examples = [{"prompt": "p p p p p", "completion": "c c"}]

    assert _validate_example_token_budget(FakeTokenizer(), examples, 8) == 8
