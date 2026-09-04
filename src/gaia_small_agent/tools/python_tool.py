from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import Any

from .base import Tool, ToolResult


_BLOCKED_NAMES = {
    "open", "eval", "exec", "compile", "__import__", "globals", "locals", "vars",
    "input", "breakpoint", "help", "memoryview",
}
_SAFE_BUILTINS = (
    "print", "range", "sum", "min", "max", "len", "abs", "round", "sorted",
    "enumerate", "zip", "any", "all", "str", "int", "float", "bool", "list",
    "dict", "tuple", "set", "frozenset", "reversed",
)


def _validate_code(code: str) -> str | None:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return f"invalid Python syntax: {exc.msg}"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "imports are not allowed"
        if isinstance(node, ast.Name) and node.id in _BLOCKED_NAMES:
            return f"blocked name: {node.id}"
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            return f"private/dunder attribute access is not allowed: {node.attr}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_NAMES:
            return f"blocked call: {node.func.id}"
    return None


def _wrapper(code: str) -> str:
    safe_pairs = ", ".join(f"{name!r}: {name}" for name in _SAFE_BUILTINS)
    return (
        f"_safe = {{{safe_pairs}}}\n"
        f"_code = {code!r}\n"
        "exec(compile(_code, '<agent-python>', 'exec'), {'__builtins__': _safe}, {})\n"
    )


class PythonTool(Tool):
    name = "python"
    description = "Run bounded, import-free Python for arithmetic and local in-memory data transformations. It cannot read files, environment variables, processes, or the network."
    schema = {
        "type": "object",
        "properties": {"code": {"type": "string", "description": "Import-free Python source; print the needed result"}},
        "required": ["code"],
        "additionalProperties": False,
    }

    def __init__(self, timeout_s: float = 10.0, max_output_chars: int = 12000):
        self.timeout_s = timeout_s
        self.max_output_chars = max_output_chars

    def run(self, arguments: dict[str, Any], workspace: Path) -> ToolResult:
        code = arguments.get("code")
        if not isinstance(code, str) or not code.strip():
            return ToolResult(False, "'code' must be a non-empty string", "BAD_ARGUMENTS")
        unsafe = _validate_code(code)
        if unsafe is not None:
            return ToolResult(False, unsafe, "UNSAFE_PYTHON")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", _wrapper(code)],
                cwd=workspace,
                text=True,
                capture_output=True,
                timeout=self.timeout_s,
                env={},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False, f"Python exceeded {self.timeout_s}s timeout", "TIMEOUT")
        output = (proc.stdout + ("\nSTDERR:\n" + proc.stderr if proc.stderr else "")).strip()
        output = output[: self.max_output_chars]
        if proc.returncode != 0:
            return ToolResult(False, output or f"Python exited {proc.returncode}", "PYTHON_ERROR")
        return ToolResult(True, output)
