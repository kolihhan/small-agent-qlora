from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import requests

from .tools import check_default_tools


_TRANSFORMERS_RUNTIME_MODULES = ("torch", "transformers", "bitsandbytes", "accelerate")


def _ollama_checks(model: str, ollama_url: str) -> tuple[dict, dict]:
    try:
        response = requests.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=5.0)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
            raise ValueError("/api/tags response is missing a models list")
        names = {
            str(row.get("name") or row.get("model") or "")
            for row in payload["models"]
            if isinstance(row, dict)
        }
        backend = {"ok": True, "detail": f"Ollama reachable at {ollama_url.rstrip('/')}"}
        if model in names:
            model_check = {"ok": True, "detail": f"model available: {model}"}
        else:
            model_check = {"ok": False, "detail": f"requested model not found: {model}"}
        return backend, model_check
    except requests.RequestException as exc:
        detail = f"Ollama unavailable: {exc}"
        return {"ok": False, "detail": detail}, {"ok": False, "detail": f"model availability not checked: {model}"}
    except (TypeError, ValueError) as exc:
        detail = f"Ollama returned an invalid model-list response: {exc}"
        return {"ok": False, "detail": detail}, {"ok": False, "detail": f"model availability not established: {model}"}


def _transformers_checks(model: str) -> tuple[dict, dict]:
    missing = [name for name in _TRANSFORMERS_RUNTIME_MODULES if find_spec(name) is None]
    if missing:
        backend = {"ok": False, "detail": "missing runtime modules: " + ", ".join(missing)}
    else:
        backend = {"ok": True, "detail": "Transformers runtime dependencies available"}
    model_check = {
        "ok": True,
        "detail": f"model identity configured as {model}; weights are not loaded by doctor",
    }
    return backend, model_check


def run_doctor(backend: str, model: str, ollama_url: str, workspace: str | Path) -> dict:
    """Return a lightweight readiness report without loading model weights."""
    if backend not in {"ollama", "transformers"}:
        raise ValueError(f"unsupported backend: {backend}")

    tool_report = check_default_tools(workspace, live_search=False)
    if backend == "ollama":
        backend_check, model_check = _ollama_checks(model, ollama_url)
    else:
        backend_check, model_check = _transformers_checks(model)

    checks = {
        "backend": backend_check,
        "model": model_check,
        "tools": {
            "ok": bool(tool_report.get("ready")),
            "detail": "default tool surface ready" if tool_report.get("ready") else "one or more default tool checks failed",
        },
    }
    ready = all(check["ok"] for check in checks.values())
    return {
        "ready": ready,
        "backend": backend,
        "model": model,
        "checks": checks,
        "tools": tool_report,
    }
