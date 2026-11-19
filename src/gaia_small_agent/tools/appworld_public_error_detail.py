"""Fail-closed PUBLIC validation-field extraction for AppWorld HTTP errors.

Never expose raw exception text, task values, tokens or evaluator information.
This function is opt-in and returns documented field names only.
"""
from __future__ import annotations

import json
import re


def documented_parameter_hint(exception_text: str, public_api_doc: dict) -> str | None:
    """Return a safe category for one documented parameter; otherwise None."""
    if not isinstance(exception_text, str) or not isinstance(public_api_doc, dict):
        return None
    match = re.match(r"^Response status code is 422:\s*\n([\s\S]*)$", exception_text)
    if match is None:
        return None
    params = public_api_doc.get("parameters")
    if not isinstance(params, list):
        return None
    documented = {
        p["name"] for p in params if isinstance(p, dict)
        and isinstance(p.get("name"), str)
        and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]{0,63}", p["name"])
    }
    if not documented:
        return None
    try:
        payload = json.loads(match.group(1))
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("detail"), list):
        return None
    for item in payload["detail"]:
        if not isinstance(item, dict):
            continue
        loc = item.get("loc")
        typ = item.get("type")
        if not isinstance(loc, list) or len(loc) < 2 or not isinstance(typ, str):
            continue
        param = loc[-1]
        if not isinstance(param, str) or param not in documented:
            continue
        if typ in ("missing", "value_error.missing"):
            return "Public API validation: required documented parameter '" + param + "' is missing."
        if typ in ("string_type", "int_type", "float_type", "bool_type",
                   "list_type", "dict_type", "type_error.integer", "type_error.str"):
            return "Public API validation: documented parameter '" + param + "' has an invalid type."
    return None
