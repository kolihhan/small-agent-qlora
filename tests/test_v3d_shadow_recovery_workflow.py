"""Confirm paired shadow workflow uses one unchanged v3D adapter."""
from pathlib import Path
from scripts import run_gaia_eval_shard


def test_shadow_and_recovery_are_opt_in():
    args = ["--hf-token", "test", "--shard-index", "0", "--work-root", "x", "--output", "y"]
    assert run_gaia_eval_shard.build_parser().parse_args(args).partition == "evaluation"
    assert not run_gaia_eval_shard.build_parser().parse_args(args).terminal_recovery
    test = run_gaia_eval_shard.build_parser().parse_args(args + ["--partition", "shadow", "--terminal-recovery"])
    assert test.partition == "shadow" and test.terminal_recovery


def test_workflow_pins_v3d_for_both_arms_and_never_uploads_gated_rows():
    flow = Path(".github/workflows/gaia-v3d-terminal-recovery-shadow25.yml").read_text()
    assert "shard: [0, 1, 2, 3, 4]" in flow
    assert "run_arm baseline" in flow and "run_arm recovery" in flow
    assert "V3B_RUN_ID" not in flow
    assert 'adapter="matched-input/v3d/qwen35-9b-qlora-v3d-adapter.gguf"' in flow
    assert "--partition shadow --backend openai_compat" in flow
    assert "--terminal-recovery" in flow
    assert "V3D_SHA256" in flow and "V3D_RUN_ID" in flow
    assert flow.index("Download frozen v3D adapter") < flow.index("Download and verify frozen model")
    for section in flow.split("- name: Upload ")[1:]:
        step = section.split("\n  - name:")[0]
        assert "path: sanitized/" in step
        assert "results.jsonl" not in step and "runs/" not in step
