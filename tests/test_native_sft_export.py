import json
from pathlib import Path


def test_native_sft_export_preserves_tools_calls_observations_and_final(tmp_path):
    from gaia_small_agent.training.native_sft import export_native_sft

    source = tmp_path / "source.jsonl"
    output = tmp_path / "native.jsonl"
    row = {
        "task_id": "case-1",
        "capability": "multi_step",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "Read a source",
                    "parameters": {
                        "type": "object",
                        "properties": {"source": {"type": "string"}},
                        "required": ["source"],
                    },
                },
            }
        ],
        "messages": [
            {"role": "system", "content": "SYSTEM RULES"},
            {"role": "user", "content": "Read note.txt"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "read", "arguments": {"source": "note.txt"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "name": "read", "content": "TOKEN=abc"},
            {"role": "assistant", "content": "abc"},
        ],
    }
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")

    summary = export_native_sft(source, output)
    exported = json.loads(output.read_text(encoding="utf-8"))
    messages = exported["messages"]

    assert summary == {"rows": 1, "assistant_actions": 2, "tool_actions": 1, "tool_observations": 1}
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "tool", "assistant"]
    assert messages[0]["content"].startswith("# Tools\n\nYou have access to the following functions:")
    assert '<tools>\n{"type": "function"' in messages[0]["content"]
    assert messages[0]["content"].endswith("SYSTEM RULES")
    assert messages[2]["content"] == (
        "<tool_call>\n"
        "<function=read>\n"
        "<parameter=source>\n"
        "note.txt\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    assert messages[3] == {"role": "tool", "content": "TOKEN=abc"}
    assert messages[4] == {"role": "assistant", "content": "abc"}


def test_native_sft_export_keeps_no_tool_answer_and_tools_visible(tmp_path):
    from gaia_small_agent.training.native_sft import export_native_sft

    source = tmp_path / "source.jsonl"
    output = tmp_path / "native.jsonl"
    row = {
        "task_id": "case-2",
        "capability": "no_tool_stop",
        "tools": [{"type": "function", "function": {"name": "python", "parameters": {"type": "object"}}}],
        "messages": [
            {"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": "Return DIRECT-7"},
            {"role": "assistant", "content": "DIRECT-7"},
        ],
    }
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")

    summary = export_native_sft(source, output)
    messages = json.loads(output.read_text(encoding="utf-8"))["messages"]

    assert summary["assistant_actions"] == 1
    assert summary["tool_actions"] == 0
    assert "<tools>" in messages[0]["content"]
    assert messages[-1] == {"role": "assistant", "content": "DIRECT-7"}
