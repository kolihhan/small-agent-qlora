from pathlib import Path


SCRIPT = Path("scripts/run_gaia_error_diagnostic.py")
WORKFLOW = Path(".github/workflows/gaia-error-diagnostic.yml")


def test_gaia_error_diagnostic_is_real_gaia_and_aggregate_only():
    script = SCRIPT.read_text(encoding="utf-8")
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert 'partition="diagnostic"' in script
    assert '"error_codes"' in script
    assert '"tool_calls_by_name"' in script
    assert '"tool_errors_by_name"' in script
    assert '"error_codes_by_tool"' in script
    assert '"task_id"' not in script[script.index("payload = {"):]
    assert '"Question"' not in script[script.index("payload = {"):]
    assert '"Final answer"' not in script[script.index("payload = {"):]

    assert "MODEL: qwen3.5:9b" in workflow
    assert 'SHARD_COUNT: "5"' in workflow
    assert "max-parallel: 5" in workflow
    assert "expected Diagnostic25 total 25" in workflow
    assert "name: gaia-9b-error-diagnostic-summary" in workflow
    upload = workflow[workflow.index("Upload final sanitized diagnostic only"):]
    assert "results.jsonl" not in upload
    assert "trace" not in upload
