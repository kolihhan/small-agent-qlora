from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_TOOL_PREAMBLE = """# Tools

You have access to the following functions:

<tools>"""

_TOOL_SUFFIX = """</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>"""


def _render_system(tools: list[dict[str, Any]], original: str) -> str:
    rendered_tools = "\n".join(json.dumps(tool, ensure_ascii=False, sort_keys=False) for tool in tools)
    prefix = f"{_TOOL_PREAMBLE}\n{rendered_tools}\n{_TOOL_SUFFIX}"
    original = original.strip()
    return prefix if not original else f"{prefix}\n\n{original}"


def _render_argument(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _render_tool_call(name: str, arguments: dict[str, Any]) -> str:
    lines = ["<tool_call>", f"<function={name}>"]
    for key, value in arguments.items():
        lines.extend(
            [
                f"<parameter={key}>",
                _render_argument(value),
                "</parameter>",
            ]
        )
    lines.extend(["</function>", "</tool_call>"])
    return "\n".join(lines)


def _assistant_content(message: dict[str, Any]) -> tuple[str, int]:
    content = str(message.get("content") or "").strip()
    tool_calls = message.get("tool_calls") or []
    rendered_calls: list[str] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            raise ValueError("assistant tool call must be an object")
        function = call.get("function") if isinstance(call.get("function"), dict) else call
        name = str(function.get("name") or "").strip()
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ValueError(f"tool call {name!r} has non-JSON string arguments") from exc
        if not name or not isinstance(arguments, dict):
            raise ValueError("tool call requires a name and object arguments")
        rendered_calls.append(_render_tool_call(name, arguments))
    if not rendered_calls:
        return content, 0
    calls_text = "\n".join(rendered_calls)
    return (f"{content}\n\n{calls_text}" if content else calls_text), len(rendered_calls)


def _convert_row(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    tools = row.get("tools") or []
    messages = row.get("messages") or []
    if not isinstance(tools, list) or not tools:
        raise ValueError("native SFT row requires visible tool schemas")
    if not isinstance(messages, list) or not messages:
        raise ValueError("native SFT row requires messages")
    if not isinstance(messages[0], dict) or messages[0].get("role") != "system":
        raise ValueError("native SFT row must start with a system message")

    converted: list[dict[str, str]] = []
    assistant_actions = tool_actions = tool_observations = 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError("message must be an object")
        role = str(message.get("role") or "")
        if role == "system":
            if index != 0:
                raise ValueError("system message must be first")
            converted.append({"role": "system", "content": _render_system(tools, str(message.get("content") or ""))})
        elif role == "user":
            converted.append({"role": "user", "content": str(message.get("content") or "")})
        elif role == "assistant":
            content, calls = _assistant_content(message)
            if not content:
                raise ValueError("assistant action is empty")
            converted.append({"role": "assistant", "content": content})
            assistant_actions += 1
            tool_actions += calls
        elif role == "tool":
            converted.append({"role": "tool", "content": str(message.get("content") or "")})
            tool_observations += 1
        else:
            raise ValueError(f"unsupported message role: {role!r}")

    return {"messages": converted}, {
        "assistant_actions": assistant_actions,
        "tool_actions": tool_actions,
        "tool_observations": tool_observations,
    }


def export_native_sft(source: str | Path, output: str | Path) -> dict[str, int]:
    """Export verified oracle rows into Qwen3.5-native role/content SFT JSONL.

    The Qwen3.5 GGUF chat template receives ordinary role/content messages. Tool
    schemas are embedded in the system message using Qwen3.5's official tool
    preamble; structured assistant tool calls are rendered in the model's native
    <tool_call>/<function>/<parameter> representation. Tool observations remain
    role=tool so the GGUF chat template renders them as <tool_response> blocks.
    """
    source = Path(source)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = assistant_actions = tool_actions = tool_observations = 0
    lines: list[str] = []
    for line_no, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("verified") is False:
            raise ValueError(f"line {line_no}: refusing unverified row")
        converted, counts = _convert_row(row)
        lines.append(json.dumps(converted, ensure_ascii=False, sort_keys=False))
        rows += 1
        assistant_actions += counts["assistant_actions"]
        tool_actions += counts["tool_actions"]
        tool_observations += counts["tool_observations"]

    if not rows:
        raise ValueError("No rows to export")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "rows": rows,
        "assistant_actions": assistant_actions,
        "tool_actions": tool_actions,
        "tool_observations": tool_observations,
    }


__all__ = ["export_native_sft"]
