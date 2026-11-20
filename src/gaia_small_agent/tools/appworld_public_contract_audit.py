"""Public-only AppWorld tool-contract diagnostics; no model-visible changes.

Static contract audit is based on PUBLIC AppWorld API docs. Runtime HTTP status
histograms contain numbers/categorical parser outcomes only; never store raw
exception text, task text, arguments, API values or evaluator details.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import json
import re
from typing import Any

from .appworld_public_error_detail import documented_parameter_hint


def public_doc_visibility_audit(world: Any) -> dict[str, int]:
    """Count public parameter fields currently omitted from _doc_info.

    This intentionally returns counts only: NEVER task-specific API names,
    example values, parameter names or source task contents.
    """
    from .appworld_tool import AppWorldDocsTool
    tool = AppWorldDocsTool(world)
    counts: Counter[str] = Counter()
    allowed = set(world.task.allowed_apps) - {"admin", "api_docs"}
    docs_by_app = world.task.api_docs
    for app in allowed:
        apis = docs_by_app.get(app, {})
        if not isinstance(apis, dict):
            counts["invalid_public_api_map"] += 1
            continue
        for api, doc in apis.items():
            counts["public_apis"] += 1
            if not isinstance(doc, dict):
                counts["invalid_public_api_doc"] += 1
                continue
            params = doc.get("parameters", [])
            if not isinstance(params, list):
                counts["invalid_public_parameter_list"] += 1
                continue
            counts["public_parameters"] += len(params)
            counts["parameters_hidden_by_first24_cap"] += max(0, len(params) - 24)
            if isinstance(doc.get("response_schemas"), dict) and doc["response_schemas"]:
                counts["public_response_schema_examples_omitted"] += 1
            for p in params[:24]:
                if not isinstance(p, dict):
                    counts["invalid_public_parameter"] += 1
                    continue
                if "default" in p:
                    counts["parameter_default_field_omitted"] += 1
                    if p["default"] is not None:
                        counts["non_null_default_values_omitted"] += 1
                if "constraints" in p:
                    counts["parameter_constraints_field_omitted"] += 1
                    if isinstance(p["constraints"], (list, dict, str)) and p["constraints"]:
                        counts["nonempty_public_constraints_omitted"] += 1
            # Measure exact CURRENT tool response; AgentRuntime will cap each
            # tool message at 2000 characters, independent of the 10k tool cap.
            result = tool.run({"app": app, "api": api}, Path("."))
            if not result.ok:
                counts["public_doc_render_failures"] += 1
                continue
            counts["actual_public_api_doc_outputs"] += 1
            current_size = len(result.content)
            counts["current_doc_visible_string_chars"] += current_size
            if current_size > 2000:
                counts["current_public_docs_exceed_agent_2000_char_cap"] += 1
            # Estimate a *separate* future opt-in that adds only nonempty
            # fields, not a new model-visible behavior in this experiment.
            full = {k: doc[k] for k in ("description", "method", "path") if k in doc}
            richer = []
            for p in params[:24]:
                if not isinstance(p, dict):
                    continue
                q = {k: p[k] for k in ("name","type","required","description") if k in p}
                if p.get("default") is not None:
                    q["default"] = p["default"]
                if isinstance(p.get("constraints"), (list, dict, str)) and p["constraints"]:
                    q["constraints"] = p["constraints"]
                richer.append(q)
            full["parameters"] = richer
            potential = json.dumps(
                {"app": app, "api": api, "public_api_doc": full},
                ensure_ascii=False, sort_keys=True, default=str)
            projected_size = len(potential)
            counts["potential_nonempty_details_chars"] += projected_size
            if projected_size > 2000:
                counts["potential_public_docs_exceed_agent_2000_char_cap"] += 1
            if current_size <= 2000 and projected_size > 2000:
                counts["additional_docs_cross_2000_char_cap_if_enriched"] += 1
    return dict(sorted(counts.items()))


def public_http_status(exception_text: str) -> int | None:
    """Parse exact upstream Requester.raise_if_failure status prefix."""
    if not isinstance(exception_text, str):
        return None
    match = re.match(r"^Response status code is ([45][0-9]{2}):\n", exception_text)
    return int(match.group(1)) if match else None


def public_422_probe_category(exception_text: str, public_api_doc: dict) -> str:
    """Finite allowlisted categories only; never return exception content."""
    status = public_http_status(exception_text)
    if status is None:
        return "UNRECOGNIZED_STATUS_PREFIX"
    if status != 422:
        return "NOT_422"
    params = public_api_doc.get("parameters") if isinstance(public_api_doc, dict) else None
    if not isinstance(params, list) or not params:
        return "PUBLIC_PARAMETER_SCHEMA_MISSING"
    if documented_parameter_hint(exception_text, public_api_doc):
        return "PUBLIC_FIELD_HINT_AVAILABLE"
    return "NO_DOCUMENTED_FIELD_HINT"
