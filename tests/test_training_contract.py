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
