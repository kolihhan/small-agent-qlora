"""Synthetic isolated check of frozen DEV48 artifact aggregation. No real questions/gold."""
import json
import pytest

from scripts.diagnose_dev48_safe_metadata import SPECS, diagnose


def _write(root, group, disjoint=True):
    root.mkdir()
    spec = SPECS[group]
    for shard in range(4):
        for arm in ("control", spec["candidate"]):
            cases=[]
            for k in range(6):
                ident = shard*6 + k + (0 if group == "dev24a" or not disjoint else 24)
                cases.append({
                    "id_sha256": f"hashed_{ident}", "hop_count": 3 if k % 2 else 4,
                    "correct": False, "completed": k % 2 == 1,
                    "stop_reason": "final" if k % 2 else "invalid_final",
                    "tool_calls": 4, "steps": 5, "tool_errors": 0,
                    "duplicate_calls": 0, "elapsed_s": 1.2,
                })
            obj={
                "protocol":spec["protocol"], "n":6,"shard":shard,"arm":arm,
                "manifest_sha256":spec["manifest"],
                "source_sha256":"15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b",
                "only_variable":spec["only_variable"],
                "adapter":"verified_qwen3.5_9b_v8_step100",
                "max_steps":12, "max_new_tokens":512,
                "answer_label_used_in_decisions":False,
                "no_task_or_answers_exported":True,
                "same_four_tools":True,
                "cases":cases,
            }
            (root / f"shard-{shard}-{arm}.json").write_text(json.dumps(obj))


def test_48_unique_control_cases_and_report_aggregate_only(tmp_path):
    a,b=tmp_path/"a",tmp_path/"b"
    _write(a,"dev24a")
    _write(b,"dev24b")
    r=diagnose(a,b)
    assert r["n_unique_tasks"] == 48
    assert r["groups"]["dev24a"]["control"]["completed_wrong"] == 12
    assert r["groups"]["dev24b"]["control"]["wrong_without_tool_error"] == 24
    assert r["groups"]["dev24b"]["control"]["stop_reasons"]["invalid_final"] == 12
    assert r["model_inference"] is False and r["answer_or_gold_exported"] is False
    assert "hashed_" not in json.dumps(r)
    assert "question" not in json.dumps(r)


def test_refuse_reused_dev24_identities(tmp_path):
    a,b=tmp_path/"a",tmp_path/"b"
    _write(a,"dev24a")
    _write(b,"dev24b",disjoint=False)
    with pytest.raises(ValueError,match="not share"):
        diagnose(a,b)


def test_refuse_manifest_drift(tmp_path):
    a,b=tmp_path/"a",tmp_path/"b"
    _write(a,"dev24a")
    _write(b,"dev24b")
    path=b/"shard-3-state.json"
    data=json.loads(path.read_text())
    data["manifest_sha256"]="wrong"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="integrity mismatch"):
        diagnose(a,b)
