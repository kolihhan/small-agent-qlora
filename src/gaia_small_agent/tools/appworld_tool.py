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
from .appworld_public_error_detail import documented_parameter_hint
from .appworld_public_contract_audit import public_422_probe_category
from .appworld_public_auth_observation import documented_auth_observation

_TOOL_APPS_DENY = frozenset({"admin", "api_docs"})
_SUPERVISOR_ALLOW = frozenset({"complete_task"})
# OFF by default. Parity is an explicit experiment-only tool *capability*
# toggle, not a default model instruction and not an authentication shortcut.
_SUPERVISOR_PARITY_APIS = frozenset({"show_profile", "show_account_passwords"})
_MAX_JSON_CHARS = 10000


def allowed_public_api(world: Any, app: str, api: str, *, official_supervisor_parity: bool = False) -> bool:
    if (not isinstance(app, str) or not isinstance(api, str)
        or not app.isidentifier() or not api.isidentifier()
        or app in _TOOL_APPS_DENY or api.startswith("_")):
        return False
    if app == "supervisor":
        if api in _SUPERVISOR_ALLOW:
            return True  # Preserve baseline supervisor.complete_task.
        # Never expose any additional Supervisor API, even with parity enabled.
        return (
            official_supervisor_parity is True
            and api in _SUPERVISOR_PARITY_APIS
            and app in set(world.task.allowed_apps)
            and api in world.task.api_docs.get(app, {})
        )
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


def _public_response_shape(example: Any, *, depth: int = 0) -> Any:
    """Show ONLY documented response structure, never generated example values.

    AppWorld public `response_schemas.success` contains generated *examples*,
    which may look like credentials. They are NOT actual passwords. Expose
    a bounded shape to the model instead of misleading fake examples.
    """
    if depth >= 4:
        return {"type": "nested"}
    if isinstance(example, dict):
        return {"type": "object", "fields": {
            str(k): _public_response_shape(v, depth=depth + 1)
            for k, v in list(sorted(example.items()))[:20]
        }}
    if isinstance(example, list):
        return {"type": "array", "items": (
            _public_response_shape(example[0], depth=depth + 1)
            if example else {"type": "unknown"}
        )}
    if isinstance(example, bool):
        return {"type": "boolean"}
    if isinstance(example, int):
        return {"type": "integer"}
    if isinstance(example, float):
        return {"type": "number"}
    if isinstance(example, str):
        return {"type": "string"}
    return {"type": "unknown"}


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

    def __init__(self, world: Any, *, official_supervisor_parity: bool = False):
        self.world = world
        self.official_supervisor_parity = official_supervisor_parity is True
        if self.official_supervisor_parity:
            # Public API *names* only; no profile/password/token values here.
            self.description = (
                self.description
                + " Supervisor publicly documents show_profile and "
                "show_account_passwords for legitimate account login."
            )

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
            # Byte-for-byte original behavior when the experiment is OFF.
            if not self.official_supervisor_parity:
                return ToolResult(True, json.dumps({
                    "apis": ["complete_task"], "note": "Use after task work is complete."
                }))
            if api:
                if not allowed_public_api(
                    self.world, app, api, official_supervisor_parity=True
                ):
                    return ToolResult(False, "No public API by this name.", "API_NOT_FOUND")
                public = self.world.task.api_docs.get(app, {}).get(api)
                if not isinstance(public, dict):
                    return ToolResult(False, "No public API by this name.", "API_NOT_FOUND")
                info = _doc_info(public)
                responses = public.get("response_schemas")
                if isinstance(responses, dict) and "success" in responses:
                    # Only the *shape* of the official public success example.
                    # Do not deliver AppWorld's fabricated example credential values.
                    info["response_shape"] = _public_response_shape(responses["success"])
                return ToolResult(True, json.dumps({
                    "app": app, "api": api, "public_api_doc": info,
                }, ensure_ascii=False, sort_keys=True)[:_MAX_JSON_CHARS])
            apis = ["complete_task"]
            apis += sorted(
                name for name in _SUPERVISOR_PARITY_APIS
                if allowed_public_api(
                    self.world, app, name, official_supervisor_parity=True
                )
            )
            return ToolResult(True, json.dumps({
                "apis": apis,
                "note": "Use Supervisor public profile/password retrieval for authorized app login; "
                        "complete_task only after task work is complete.",
            }, sort_keys=True))
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

    def __init__(self, world: Any, max_calls: int = 50, *, report_http_failures: bool = False,
                 public_422_field_hint: bool = False,
                 capture_422_field_hint: bool = False,
                 public_401_auth_hint: bool = False,
                 capture_401_auth_hint: bool = False,
                 official_supervisor_parity: bool = False):
        self.world = world
        self.official_supervisor_parity = official_supervisor_parity is True
        self.max_calls = max_calls
        self.report_http_failures = report_http_failures
        self.public_422_field_hint = public_422_field_hint
        self.capture_422_field_hint = capture_422_field_hint
        self.public_401_auth_hint = public_401_auth_hint
        self.capture_401_auth_hint = capture_401_auth_hint
        self.public_401_auth_candidates: list[dict] = []
        self.auth_401_public_category_counts: dict[str,int] = {}
        self.public_422_candidates: list[dict] = []
        self.model_step_counter = None
        self.calls = 0
        self.errors = 0
        self.http_errors = 0
        # Safe aggregated diagnostic counters; never store raw exception text.
        self.http_status_counts: dict[str, int] = {}
        self.http_422_probe_outcomes: dict[str, int] = {}
        # Strictly safe per-step telemetry; never retain API arguments or responses.
        self.http_error_events: list[dict[str, int | None]] = []

    def run(self, arguments: dict, workspace: Path) -> ToolResult:
        app = arguments.get("app")
        api = arguments.get("api")
        data = arguments.get("arguments")
        if not isinstance(data, dict) or not allowed_public_api(
            self.world, app, api,
            official_supervisor_parity=self.official_supervisor_parity
        ):
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
                    status_key = status.group(1)
                    # Capture only the upstream numeric HTTP status and model step;
                    # the exception string and API values remain process-private.
                    counter = self.model_step_counter
                    self.http_error_events.append({
                        "model_step":counter() if callable(counter) else None,
                        "api_call_index":self.calls,
                        "http_status":int(status_key),
                    })
                    self.http_status_counts[status_key] = (
                        self.http_status_counts.get(status_key, 0) + 1)
                    if status_key == "422":
                        public_doc = self.world.task.api_docs.get(app, {}).get(api, {})
                        category = public_422_probe_category(str(exc), public_doc)
                        self.http_422_probe_outcomes[category] = (
                            self.http_422_probe_outcomes.get(category, 0) + 1)
                    visible = ("AppWorld API rejected this request (HTTP " +
                               status.group(1) + "). Check the API documentation and parameters.")
                    if status_key == "401":
                        apis = self.world.task.api_docs.get(app, {})
                        doc401 = apis.get(api, {}) if isinstance(apis, dict) else {}
                        auth401 = documented_auth_observation(
                            401, doc401, data,
                            isinstance(apis, dict) and "login" in apis)
                        if auth401 is not None:
                            code, hint401 = auth401
                            self.auth_401_public_category_counts[code] = (
                                self.auth_401_public_category_counts.get(code, 0) + 1)
                            if self.capture_401_auth_hint:
                                counter = self.model_step_counter
                                self.public_401_auth_candidates.append({
                                    "api_call_index": self.calls,
                                    "model_step": counter() if callable(counter) else None,
                                    "safe_category": code,
                                    "public_hint": hint401,
                                })
                            if self.public_401_auth_hint:
                                visible += " " + hint401
                    if self.public_422_field_hint or self.capture_422_field_hint:
                        doc = self.world.task.api_docs.get(app, {}).get(api, {})
                        hint = documented_parameter_hint(str(exc), doc)
                        if hint is not None:
                            if self.capture_422_field_hint:
                                count = self.model_step_counter
                                self.public_422_candidates.append({
                                    "api_call_index": self.calls,
                                    "model_step": count() if callable(count) else None,
                                    "hint": hint,
                                    "http_status": int(status.group(1)),
                                })
                            if self.public_422_field_hint:
                                visible += " " + hint
                    return ToolResult(False, visible, "HTTP_ERROR")
            return ToolResult(False, "AppWorld request error: "+type(exc).__name__,
                              "APPWORLD_API_ERROR")
