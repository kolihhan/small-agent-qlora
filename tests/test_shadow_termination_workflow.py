from pathlib import Path

import pytest

from scripts.analyze_termination_shift import summarize
from scripts.merge_termination_shards import merge


def _row(key, *, completed=True, correct=True, steps=2):
    return {
        "task_id": key, "completed": completed, "correct": correct,
        "steps": steps, "stop_reason": "final" if completed else "invalid_final",
        "tool_calls": 1, "duplicate_calls_blocked": 0, "tool_errors": 0, "trace": [],
    }


def test_merge_five_aggregate_only_shards():
    shards = []
    for i in range(5):
        b = {f"hidden-{i}-{j}": _row(f"hidden-{i}-{j}") for j in range(5)}
        d = {f"hidden-{i}-{j}": _row(
            f"hidden-{i}-{j}", completed=j != 0, correct=j != 0,
            steps=12 if j == 0 else 3,
        ) for j in range(5)}
        shards.append(summarize(b, d))

    final = merge(shards)
    assert final["n"] == 25 and final["shards"] == 5
    assert final["v3b"]["completed"] == 25
    assert final["v3d"]["completed"] == 20
    assert final["paired_completion"]["v3b_only"] == 5
    assert final["paired_completion"]["v3d_only"] == 0
    assert sum(final["cohorts_by_v3b_steps"][b]["n"] for b in
               ("01-03", "04-06", "07-09", "10-12")) == 25
    assert not any("hidden-" in str(part) for part in final.values())


def test_merge_rejects_incomplete_or_wrong_shard_count():
    sh = summarize({"a": _row("a")}, {"a": _row("a")})
    with pytest.raises(ValueError, match="unexpected shard count"):
        merge([sh])
    with pytest.raises(ValueError, match="unexpected aggregate population"):
        merge([sh] * 5)


def test_shadow_workflow_uploads_only_sanitized_aggregate():
    workflow = Path(".github/workflows/gaia-shadow25-termination.yml").read_text(encoding="utf-8")
    assert "branches:" in workflow
    assert "research/termination-shift-diagnostic-20261112" in workflow
    assert 'shard: [0, 1, 2, 3, 4]' in workflow
    assert "max-parallel: 5" in workflow
    assert "run_arm qlora-v3b" in workflow
    assert "run_arm qlora-v3d" in workflow
    assert "run_arm base" not in workflow
    assert "--partition shadow" in workflow
    assert "--shard-count 5" in workflow
    assert "--expected 5" in workflow
    assert "MODEL_SHA256" in workflow and "V3B_SHA256" in workflow and "V3D_SHA256" in workflow
    assert workflow.index("Verify GAIA credential") < workflow.index("Download and verify frozen model")
    # No trace, question, result.jsonl, model output, or workspace in any upload step.
    for section in workflow.split("- name: Upload ")[1:]:
        upload_step = section.split("\n  - name:")[0]
        assert "path: sanitized/" in upload_step
        for forbidden in ("results.jsonl", "runs/", "trace", "question", "workspace"):
            assert forbidden not in upload_step
