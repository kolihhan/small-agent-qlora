"""Contract checks for workflows retained in the reproducible final repository.

Historical experiments and their YAML were intentionally removed in the cleanup.
Keep contracts for the current GAIA matched evaluation and its runner only.
"""
from pathlib import Path

MATCHED = Path(".github/workflows/gaia-v3d-clean-same-runner.yml")
RUNNER = Path("scripts/run_gaia_eval_shard.py")


def test_gaia_shard_runner_preflight_is_offline():
    runner = RUNNER.read_text(encoding="utf-8")
    assert "with tempfile.TemporaryDirectory" in runner
    assert "preflight_eval_tools(workspace)" in runner
    assert "check_default_tools(workspace, live_search=False)" in runner
    assert "preflight_default_tools" not in runner


def test_matched_gaia100_keeps_frozen_protocol():
    workflow = MATCHED.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    assert 'FROZEN_EVAL_COMMIT: "08fbcaa5ed82c5238188c45d5748993bae9bf1e1"' in workflow
    assert "--ctx-size 16384 --threads 4 --parallel 1" in workflow
    assert "--reasoning-budget 0 --temperature 0" in workflow
    assert "max-parallel: 20" in workflow
    assert "shard: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]" in workflow
    assert "max_steps=12" in runner
    assert "max_new_tokens=512" in runner


def test_matched_gaia100_paired_order_and_promotion_metrics():
    workflow = MATCHED.read_text(encoding="utf-8")
    assert workflow.index("run_arm base\n") < workflow.index("run_arm qlora-v3b\n") < workflow.index("run_arm qlora-v3d\n")
    assert 'if len(rows) != 100:' in workflow
    assert 'if not (ids_by_arm["base"] == ids_by_arm["qlora-v3b"] == ids_by_arm["qlora-v3d"]):' in workflow
    assert '"completion_rate_pp"' in workflow
    assert '"duplicate_calls_blocked"' in workflow
    assert "name: matched-gaia-v3d-summary" in workflow
