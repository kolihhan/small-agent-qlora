import json
import pytest

from scripts.eval_step100_stop_transfer_musique24 import (
    FROZEN_TASKS_SHA256, PROTOCOL, frozen_tasks,
)
from scripts.score_step100_stop_transfer_musique24 import aggregate


def _write_evidence(root, *, rescues=(6, 18, 19), regressions=()):
    root.mkdir(parents=True, exist_ok=True)
    for shard in range(4):
        for arm in ("control", "stop"):
            cases = []
            for index in range(shard * 6, (shard + 1) * 6):
                base = index < 6
                success = base if arm == "control" else (base or index in rescues) and index not in regressions
                cases.append({
                    "id_sha256": f"{index:064x}",
                    "hop_count": 3 if index < 12 else 4,
                    "correct": bool(success),
                    "completed": True,
                    "stop_reason": "final",
                    "steps": 3, "tool_calls": 3,
                    "tool_errors": 0, "duplicate_calls_blocked": 0,
                    "elapsed_s": 0.5,
                })
            payload = {
                "protocol": PROTOCOL,
                "shard": shard, "arm": arm, "n_tasks": 6,
                "frozen_tasks_sha256": FROZEN_TASKS_SHA256,
                "v8_training_parent_overlap": 0,
                "tool_names": ["search", "read", "inspect", "python"],
                "max_steps": 12, "max_new_tokens": 512,
                "only_variable": "evidence_stop_contract",
                "no_task_content_or_answers_exported": True,
                "cases": cases,
            }
            (root / f"shard-{shard}-{arm}.json").write_text(json.dumps(payload))


def test_preregistered_transfer_gate_only_on_real_paired_rescue_and_cost(tmp_path):
    _write_evidence(tmp_path)
    result = aggregate(tmp_path)
    assert result["task_count"] == 24
    assert result["control_correct"] == 6
    assert result["stop_correct"] == 9
    assert result["rescues"] == 3 and result["regressions"] == 0
    assert result["transfer_gate_pass"]
    assert not result["promotion_to_main_or_GAIA_claim"]


def test_one_regression_and_weak_gain_fails_predeclared_gate(tmp_path):
    _write_evidence(tmp_path, rescues=(6, 18), regressions=(0,))
    result = aggregate(tmp_path)
    assert result["net_gain"] == 1
    assert not result["transfer_gate_pass"]


def test_duplicate_task_identity_rejected(tmp_path):
    _write_evidence(tmp_path)
    path = tmp_path / "shard-1-stop.json"
    d = json.loads(path.read_text())
    d["cases"][0]["id_sha256"] = d["cases"][1]["id_sha256"]
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="Incomplete/invalid"):
        aggregate(tmp_path)


def test_cannot_substitute_arbitrary_tuned_tasks(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    report = tmp_path / "report.json"
    tasks.write_text('{"task_id":"any"}\n')
    report.write_text("{}")
    with pytest.raises(ValueError, match="SHA mismatch"):
        frozen_tasks(tasks, report)
