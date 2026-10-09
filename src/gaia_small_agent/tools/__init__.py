import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .inspect import InspectTool
from .python_tool import PythonTool
from .read import ReadTool
from .search import SearchTool


_DEFAULT_TOOL_NAMES = ["search", "read", "inspect", "python"]
_DEPENDENCIES = ("requests", "ddgs", "pypdf", "openpyxl")


def default_tools():
    return [SearchTool(), ReadTool(), InspectTool(), PythonTool()]


def check_default_tools(workspace: str | Path, *, live_search: bool = False) -> dict:
    """Check the default tool surface without requiring model inference.

    Readiness checks validate the search dependency deterministically by default.
    A caller may opt into a live search smoke for an explicit network diagnostic,
    but benchmark startup must not depend on a transient search result.
    """
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    marker = "p4-tool-preflight"
    (workspace / "preflight.txt").write_text(marker, encoding="utf-8")

    tools = default_tools()
    names = [tool.name for tool in tools]
    dependencies: dict[str, str | None] = {}
    for package in _DEPENDENCIES:
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = None

    checks: dict[str, dict] = {}
    smokes: dict[str, dict] = {}
    surface_ok = names == _DEFAULT_TOOL_NAMES
    tool_by_name = {tool.name: tool for tool in tools}
    arguments = {
        "search": {"query": "OpenAI official", "max_results": 1},
        "read": {"source": "preflight.txt"},
        "inspect": {"path": "preflight.txt"},
        "python": {"code": "print(6 * 7)"},
    }

    for name in _DEFAULT_TOOL_NAMES:
        tool = tool_by_name.get(name)
        if tool is None:
            checks[name] = {"ok": False, "detail": "tool missing from default surface", "error_code": "TOOL_MISSING"}
            continue
        if name == "search" and not live_search:
            ok = dependencies.get("ddgs") is not None
            checks[name] = {
                "ok": ok,
                "detail": "dependency available" if ok else "ddgs dependency missing",
                "error_code": None if ok else "MISSING_DEPENDENCY",
            }
            continue
        result = tool.run(arguments[name], workspace)
        ok = bool(result.ok)
        detail = result.content
        error_code = result.error_code
        if ok and name == "read" and marker not in result.content:
            ok = False
            detail = "read preflight returned unexpected content"
            error_code = "UNEXPECTED_OUTPUT"
        if ok and name == "python" and result.content.strip() != "42":
            ok = False
            detail = "python preflight returned unexpected content"
            error_code = "UNEXPECTED_OUTPUT"
        checks[name] = {"ok": ok, "detail": detail, "error_code": error_code}
        if ok:
            smokes[name] = {
                "content_chars": len(result.content),
                "content_sha256": hashlib.sha256(result.content.encode()).hexdigest(),
            }

    dependencies_ok = all(dependencies.get(package) is not None for package in _DEPENDENCIES)
    ready = surface_ok and dependencies_ok and all(checks.get(name, {}).get("ok") is True for name in _DEFAULT_TOOL_NAMES)
    return {
        "ready": ready,
        "tool_names": names,
        "dependencies": dependencies,
        "checks": checks,
        "smokes": smokes,
    }


def preflight_default_tools(workspace: str | Path) -> dict:
    """Validate the evaluation tool surface without a flaky live-web dependency."""
    report = check_default_tools(workspace, live_search=False)
    names = report["tool_names"]
    if names != _DEFAULT_TOOL_NAMES:
        raise RuntimeError(f"diagnostic tool surface mismatch: {names}")

    for name in _DEFAULT_TOOL_NAMES:
        check = report["checks"][name]
        if check["ok"]:
            continue
        error_code = check.get("error_code")
        detail = check.get("detail") or "preflight failed"
        if error_code:
            raise RuntimeError(f"{name} preflight failed: {error_code}: {detail}")
        raise RuntimeError(f"{name} preflight failed: {detail}")

    for package in _DEPENDENCIES:
        if report["dependencies"].get(package) is None:
            raise RuntimeError(f"diagnostic dependency missing: {package}")

    return {
        "tool_names": report["tool_names"],
        "dependencies": report["dependencies"],
        "smokes": report["smokes"],
    }


__all__ = [
    "SearchTool",
    "ReadTool",
    "InspectTool",
    "PythonTool",
    "default_tools",
    "check_default_tools",
    "preflight_default_tools",
]
