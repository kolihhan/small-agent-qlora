from pathlib import Path


def test_full_pipeline_script_has_all_frozen_stages_and_summary_contract():
    text = (Path(__file__).parents[1] / "scripts" / "run_p4_full_pipeline.ps1").read_text(encoding="utf-8")
    required = [
        "diagnostic25", "generate-policy-tasks", "collect-trajectories", "train-qlora",
        "evaluation-base", "evaluation-lora", "compare-evals", "pipeline-summary.json",
        "failure_signals", "protected-question-hashes.json",
    ]
    for token in required:
        assert token in text
    assert "Gate B = FALSE" not in text
    assert "NO_QLORA_NOT_JUSTIFIED" not in text


def test_full_pipeline_preserves_incomplete_summary_and_can_resume_same_run_root():
    text = (Path(__file__).parents[1] / "scripts" / "run_p4_full_pipeline.ps1").read_text(encoding="utf-8")
    assert "status = 'incomplete'" in text
    assert "pipeline-summary.incomplete-" in text
    assert "Write-IncompleteSummary" in text


def test_full_pipeline_archives_attempt_logs_resources_and_comparison_before_overwrite():
    text = (Path(__file__).parents[1] / "scripts" / "run_p4_full_pipeline.ps1").read_text(encoding="utf-8")
    assert "Archive-ExistingAttemptFile" in text
    assert "Archive-ExistingAttemptFile $stdout" in text
    assert "Archive-ExistingAttemptFile $stderr" in text
    assert "Archive-ExistingAttemptFile $resource" in text
    assert "Archive-ExistingAttemptFile $ComparisonPath" in text


def test_full_pipeline_requires_meaningful_clean_training_set_before_qlora():
    text = (Path(__file__).parents[1] / "scripts" / "run_p4_full_pipeline.ps1").read_text(encoding="utf-8")
    assert "$MinVerifiedTrajectories = 32" in text
    for tool in ("read", "inspect", "python"):
        assert f"'{tool}'" in text
    assert "QLoRA will not start" in text
