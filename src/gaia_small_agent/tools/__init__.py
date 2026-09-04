import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .inspect import InspectTool
from .python_tool import PythonTool
from .read import ReadTool
from .search import SearchTool


def default_tools():
    return [SearchTool(), ReadTool(), InspectTool(), PythonTool()]


def preflight_default_tools(workspace: str | Path) -> dict:
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    marker = "p4-tool-preflight"
    (workspace / "preflight.txt").write_text(marker, encoding="utf-8")
    tools = default_tools()
    names = [tool.name for tool in tools]
    expected = ["search", "read", "inspect", "python"]
    if names != expected:
        raise RuntimeError(f"diagnostic tool surface mismatch: {names}")
    arguments = {
        "search": {"query": "OpenAI official", "max_results": 1},
        "read": {"source": "preflight.txt"},
        "inspect": {"path": "preflight.txt"},
        "python": {"code": "print(6 * 7)"},
    }
    smokes = {}
    for tool in tools:
        result = tool.run(arguments[tool.name], workspace)
        if not result.ok:
            raise RuntimeError(f"{tool.name} preflight failed: {result.error_code}: {result.content}")
        if tool.name == "read" and marker not in result.content:
            raise RuntimeError("read preflight returned unexpected content")
        if tool.name == "python" and result.content.strip() != "42":
            raise RuntimeError("python preflight returned unexpected content")
        smokes[tool.name] = {
            "content_chars": len(result.content),
            "content_sha256": hashlib.sha256(result.content.encode()).hexdigest(),
        }
    dependencies = {}
    for package in ("requests", "ddgs", "pypdf", "openpyxl"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError as exc:
            raise RuntimeError(f"diagnostic dependency missing: {package}") from exc
    return {"tool_names": names, "dependencies": dependencies, "smokes": smokes}


__all__ = ["SearchTool", "ReadTool", "InspectTool", "PythonTool", "default_tools", "preflight_default_tools"]
