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


def allowed_public_api(world: Any, app: str, api: str, *,
                       allow_supervisor_public_reads: bool = False) -> bool:
    if (not isinstance(app, str) or not isinstance(api, str)
        or not app.isidentifier() or not api.isidentifier()
        or app in _TOOL_APPS_DENY or api.startswith("_")):
        return False
    if app == "supervisor":
        if api in _SUPERVISOR_ALLOW:
            return True
        # Optional, strictly PUBLIC API-doc-defined GET endpoints only. Never
        # expose private supervisor or evaluator APIs, nor arbitrary writes.
        doc=world.task.api_docs.get("supervisor", {}).get(api)
        return bool(allow_supervisor_public_reads and isinstance(doc,dict)
                    and str(doc.get("method","")).lower()=="get")
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

    def __init__(self, world: Any, *, allow_supervisor_public_reads: bool = False):
        self.world = world
        self.allow_supervisor_public_reads = allow_supervisor_public_reads
        # Source-only counters: never alter the tool response seen by the model.
        self.root_discovery_calls = 0
        self.supervisor_list_calls = 0
        self.supervisor_detail_calls = 0

    def run(self, arguments: dict, workspace: Path) -> ToolResult:
        app = arguments.get("app")
        api = arguments.get("api")
        if not app:
            self.root_discovery_calls += 1
            apps = sorted(set(self.world.task.allowed_apps) - _TOOL_APPS_DENY)
            response = {
                "task_apps": apps,
                "supervisor_completion": "supervisor.complete_task"
            }
            if self.allow_supervisor_public_reads:
                # Discovery must reach the student; an allowed but
                # undiscoverable API is not an actionable intervention.
                # These are task-supplied PUBLIC GET API names ONLY: never
                # private/hidden evaluator, admin or supervisor writes.
                reads = sorted(
                    name for name in self.world.task.api_docs.get("supervisor", {})
                    if allowed_public_api(
                        self.world, "supervisor", name,
                        allow_supervisor_public_reads=True)
                    and name != "complete_task"
                )
                response["supervisor_public_read_apis"] = reads
                response["public_read_help"] = (
                    "Public supervisor GET APIs can provide permitted task "
                    "context. Inspect their parameters using "
                    "appworld_docs(app='supervisor', api=...)."
                )
            return ToolResult(True,json.dumps(response,sort_keys=True))
        if not isinstance(app, str) or app in _TOOL_APPS_DENY:
            return ToolResult(False, "App unavailable for this task.", "APP_DENIED")
        if app == "supervisor":
            if api:
                self.supervisor_detail_calls += 1
            else:
                self.supervisor_list_calls += 1
            extra=sorted(
                name for name,doc in self.world.task.api_docs.get("supervisor",{}).items()
                if allowed_public_api(self.world,"supervisor",name,
                        allow_supervisor_public_reads=self.allow_supervisor_public_reads)
                and name!="complete_task")
            if api:
                if not allowed_public_api(self.world,"supervisor",api,
                        allow_supervisor_public_reads=self.allow_supervisor_public_reads):
                    return ToolResult(False, "Undocumented or disallowed supervisor API.",
                                      "API_NOT_FOUND")
                doc=self.world.task.api_docs.get("supervisor",{}).get(api,{})
                return ToolResult(True,json.dumps({"app":"supervisor","api":api,
                        "public_api_doc":_doc_info(doc)},ensure_ascii=False,sort_keys=True))
            return ToolResult(True, json.dumps({
                "apis": sorted([*extra,"complete_task"]),
                "note": "Only officially documented public read APIs are accessible."
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

    def __init__(self, world: Any, max_calls: int = 50, *,
                 report_http_failures: bool = False,
                 allow_supervisor_public_reads: bool = False):
        self.world = world
        self.max_calls = max_calls
        self.report_http_failures = report_http_failures
        self.allow_supervisor_public_reads = allow_supervisor_public_reads
        self.calls = 0
        self.errors = 0
        self.http_errors = 0
        self.http_status_counts: dict[str, int] = {}

    def run(self, arguments: dict, workspace: Path) -> ToolResult:
        app = arguments.get("app")
        api = arguments.get("api")
        data = arguments.get("arguments")
        if not isinstance(data, dict) or not allowed_public_api(
                self.world, app, api,
                allow_supervisor_public_reads=self.allow_supervisor_public_reads):
            return ToolResult(False, "API or arguments not available for this task.", "API_DENIED")
        if self.calls >= self.max_calls:
            return ToolResult(False, "AppWorld evaluation API call budget exhausted.", "CALL_LIMIT")
        self.calls += 1
        try:
            result = self.world.requester.request(
                _app_name=app, _api_name=api, show=False,
                raise_on_failure=self.report_http_failures, **data
            )
            content = json.dumps(result, ensure_ascii=False, sort_keys=True, default=str)
            return ToolResult(True, content[:_MAX_JSON_CHARS])
        except Exception as exc:
            self.errors += 1
            if self.report_http_failures:
                # AppWorld raises Exception("Response status code is NNN: ...")
                # on HTTP failures. Only disclose the status class to Qwen;
                # never export request/response bodies or hidden task values.
                import re
                status = re.search(r"(?:status code is|response status code is|status code)\s*(\d{3})",str(exc),re.I)
                if status is not None:
                    self.http_errors += 1
                    code = status.group(1)
                    self.http_status_counts[code] = self.http_status_counts.get(code, 0) + 1
                    return ToolResult(
                        False,
                        "AppWorld API rejected this request (HTTP "+status.group(1)+
                        "). Check the API documentation and parameters.",
                        "HTTP_ERROR")
            return ToolResult(False, "AppWorld request error: "+type(exc).__name__,
                              "APPWORLD_API_ERROR")
