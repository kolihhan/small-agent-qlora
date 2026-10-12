import json

from scripts.run_gaia_error_diagnostic import _aggregate_trace_counts


def test_aggregate_trace_counts_preserves_unicode_line_separator_inside_json_string(tmp_path):
    results = tmp_path / "results.jsonl"
    record = {
        "trace": [
            {
                "kind": "tool_result",
                "data": {
                    "name": "read",
                    "ok": False,
                    "error_code": "READ_ERROR",
                    "content": "before\u2028after",
                },
            }
        ]
    }
    results.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    diagnostics = _aggregate_trace_counts(results)

    assert diagnostics["error_codes"] == {"READ_ERROR": 1}
    assert diagnostics["tool_errors_by_name"] == {"read": 1}
