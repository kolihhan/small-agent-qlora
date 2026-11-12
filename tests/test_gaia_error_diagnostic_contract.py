from pathlib import Path


SCRIPT = Path("scripts/run_gaia_error_diagnostic.py")


def test_gaia_error_diagnostic_script_sanitizes_aggregates():
    script = SCRIPT.read_text(encoding="utf-8")

    assert 'partition="diagnostic"' in script
    assert '"error_codes"' in script
    assert '"tool_calls_by_name"' in script
    assert '"tool_errors_by_name"' in script
    assert '"error_codes_by_tool"' in script
    assert 'event.get("data")' in script
    assert 'event.get("details")' not in script
    assert '"task_id"' not in script[script.index("payload = {"):]
    assert '"Question"' not in script[script.index("payload = {"):]
    assert '"Final answer"' not in script[script.index("payload = {"):]

