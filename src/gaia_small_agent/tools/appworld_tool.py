"""Original AppWorld task-scoped API tool bridge for optional closed-loop evaluation.

All API schema/doc details are from PUBLIC AppWorld docs visible to the agent.
No official reference code, answer, state evaluator, or ground-truth requests
are available to tool clients. Code runs only in the isolated AppWorld local
simulation, with an explicit allowlist of task apps and one supervisor final API.

This is a *separate* tool environment from historical GAIA search/read/python,
and therefore does not prove transfer to GAIA.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import Tool, ToolResult

_TOOL_APPS_DENY = frozenset({"admin", "api_docs"})
_SUPERVISOR_ALLOW = frozenset({"complete_task"})
_MAX_JSON_CHARS = 10000


def allowed_public_api(world: Any, app: str, api: str) -> bool:
    if (not isinstance(app, str) or not isinstance(api, str)
        or not app.isidentifier() or not api.isidentifier()
        or app in _TOOL_APPS_DENY or api.startswith("_")):
        return False
    if app == "supervisor":
        return api in _SUPERVISOR_ALLOW
    return app in set(world.task.allowed_apps) and api in world.task.api_docs.get(app, {})


def _doc_info(doc: dict) -> dict:
    # Public human-visible documentation only; no hidden solution data.
    out = {k: doc[k] for k in ("description", "method", "path") if k in doc}
    params = doc.get("parameters")
    if isinstance(params, list):
        out["parameters"] = [{
            k: p[k] for k in ("name", "type", "required", "description")
            if k in p
        } for p in params[:24] if isinstance(p, dict)]
    return out


class AppWorldDocsTool(Tool):
    name = "appworld_docs"
    description = (
        "List available AppWorld app APIs, or inspect public parameter docs for "
        "one API. Use this before calling an API whose arguments are unknown."
    )
    schema = {
        "type": "object",
        "properties": {
            "app": {"type": "string", "description": "App name; omit to list available apps."},
            "api": {"type": "string", "description": "API name to inspect after choosing app."},
        },
        "additionalProperties": False,
    }

    def __init__(self, world: Any):
        self.world = world

    def run(self, arguments: dict, workspace: Path) -> ToolResult:
        app = arguments.get("app")
        api = arguments.get("api")
        if not app:
            apps = sorted(set(self.world.task.allowed_apps) - _TOOL_APPS_DENY)
            return ToolResult(True, json.dumps({
                "task_apps": apps,
                "supervisor_completion": "supervisor.complete_task"
            }, sort_keys=True))
        if not isinstance(app, str) or app in _TOOL_APPS_DENY:
            return ToolResult(False, "App unavailable for this task.", "APP_DENIED")
        if app == "supervisor":
            return ToolResult(True, json.dumps({
                "apis": ["complete_task"], "note": "Use after task work is complete."
            }))
        if app not in set(self.world.task.allowed_apps):
            return ToolResult(False, "App unavailable for this task.", "APP_DENIED")
        docs = self.world.task.api_docs.get(app, {})
        if api:
            if not allowed_public_api(self.world, app, api):
                return ToolResult(False, "No public API by this name.", "API_NOT_FOUND")
            info = {"app": app, "api": api, "public_api_doc": _doc_info(docs[api])}
        else:
            info = {"app": app, "apis": sorted(docs.keys())}
        return ToolResult(True, json.dumps(info, ensure_ascii=False, sort_keys=True)[:_MAX_JSON_CHARS])


class AppWorldCallTool(Tool):
    name = "appworld_call"
    description = (
        "Execute a real API against the CURRENT isolated AppWorld task. "
        "Specify app, API, and documented named arguments. "
        "Only task-allowed public APIs and supervisor.complete_task are permitted."
    )
    schema = {
        "type": "object",
        "properties": {
            "app": {"type": "string"},
            "api": {"type": "string"},
            "arguments": {"type": "object"},
        },
        "required": ["app", "api", "arguments"],
        "additionalProperties": False,
    }

    def __init__(self, world: Any, max_calls: int = 50):
        self.world = world
        self.max_calls = max_calls
        self.calls = 0
        self.errors = 0

    def run(self, arguments: dict, workspace: Path) -> ToolResult:
        app = arguments.get("app")
        api = arguments.get("api")
        data = arguments.get("arguments")
        if not isinstance(data, dict) or not allowed_public_api(self.world, app, api):
            return ToolResult(False, "API or arguments not available for this task.", "API_DENIED")
        if self.calls >= self.max_calls:
            return ToolResult(False, "AppWorld evaluation API call budget exhausted.", "CALL_LIMIT")
        self.calls += 1
        try:
            result = self.world.requester.request(
                _app_name=app, _api_name=api, show=False, raise_on_failure=False, **data
            )
            content = json.dumps(result, ensure_ascii=False, sort_keys=True, default=str)
            return ToolResult(True, content[:_MAX_JSON_CHARS])
        except Exception as exc:
            self.errors += 1
            # The model may see the short exception *class*, but public CI
            # evidence and external logs never include raw task or API inputs.
            return ToolResult(False, "AppWorld request error: " + type(exc).__name__, "APPWORLD_API_ERROR")
