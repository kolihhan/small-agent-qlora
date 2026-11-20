"""Public-only AppWorld HTTP 401 authentication observation hypotheses.

Only inspect documented API parameter contracts and whether an access_token
key was supplied. Never expose credentials, arguments, body, app names, task
instructions, or upstream raw error text. Disabled in production by default.
"""
from __future__ import annotations

from typing import Any


def documented_auth_observation(
    status_code: int,
    public_api_doc: dict,
    public_api_arguments: dict,
    public_app_has_login: bool,
) -> tuple[str, str] | None:
    """Return (safe reason category, safe feedback) for only grounded HTTP401.

    The public AppWorld upstream authentication handler says 401 may mean the
    endpoint/token is missing, invalid or expired; optional private access
    can reject a supplied bearer token even when a public version exists.
    A 401 alone never proves which cause actually occurred.
    """
    if status_code != 401 or not isinstance(public_api_doc, dict):
        return None
    params=public_api_doc.get("parameters")
    if not isinstance(params,list) or not isinstance(public_api_arguments,dict):
        return None
    access=[p for p in params if isinstance(p,dict)
            and p.get("name")=="access_token"
            and isinstance(p.get("required"),bool)]
    if len(access)!=1:
        return None
    required=access[0]["required"]
    supplied=bool(public_api_arguments.get("access_token"))
    if required and not supplied:
        text=("Public API authentication: the documented 'access_token' "
              "parameter is required but absent. Use this app's documented "
              "authentication flow to obtain a real token; do not guess a token.")
        if public_app_has_login:
            text+= " This app has a public login API; inspect its documentation."
        return ("REQUIRED_ACCESS_TOKEN_ABSENT",text)
    if required and supplied:
        return ("REQUIRED_ACCESS_TOKEN_PRESENT_REJECTED",
                "Public API authentication: the supplied access_token was rejected "
                "(it may be invalid, expired, or unauthorized for this endpoint). "
                "Check the app's documented login/auth flow; do not invent credentials.")
    if not required and supplied:
        return ("OPTIONAL_TOKEN_SUPPLIED_REJECTED",
                "Public API authentication: the optional access_token was rejected. "
                "Check whether the documented public version can be accessed without "
                "a token, or obtain a valid token through the public auth flow.")
    return None
