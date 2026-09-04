from __future__ import annotations

import ipaddress
import json
import re
import socket
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests

from .base import Tool, ToolResult
from .content_type import detect_content_type


def _strip_html(text: str) -> str:
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", unescape(text)).strip()


def _public_url_error(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return "Only public http(s) URLs are allowed"
    hostname = parsed.hostname.casefold()
    if hostname == "localhost" or hostname.endswith(".localhost") or hostname.endswith(".local"):
        return "Local/private URLs are blocked"
    try:
        addresses = [hostname] if _is_ip_literal(hostname) else [
            item[4][0] for item in socket.getaddrinfo(hostname, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
        ]
    except OSError as exc:
        return f"URL hostname resolution failed: {exc}"
    for address in addresses:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            return "URL hostname did not resolve to a valid IP address"
        if not ip.is_global:
            return "Local/private URLs are blocked"
    return None


def _is_ip_literal(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def _get_public_url(url: str, *, timeout: float = 20.0, max_redirects: int = 5):
    current = url
    for _ in range(max_redirects + 1):
        error = _public_url_error(current)
        if error:
            return None, error
        try:
            response = requests.get(
                current,
                timeout=timeout,
                headers={"User-Agent": "small-agent-qlora/0.1"},
                allow_redirects=False,
            )
        except Exception as exc:
            return None, f"HTTP read failed: {exc}"
        if response.is_redirect or response.is_permanent_redirect:
            location = response.headers.get("location")
            if not location:
                return None, "HTTP redirect missing Location header"
            current = urljoin(current, location)
            continue
        try:
            response.raise_for_status()
        except Exception as exc:
            return None, f"HTTP read failed: {exc}"
        return response, None
    return None, f"HTTP redirect limit exceeded ({max_redirects})"


class ReadTool(Tool):
    name = "read"
    description = "Read a public URL or a file inside the workspace. Supports common text plus PDF/XLSX when optional dependencies are installed."
    schema = {
        "type": "object",
        "properties": {
            "source": {"type": "string", "description": "URL or workspace-relative file path"},
            "max_chars": {"type": "integer", "minimum": 500, "maximum": 30000},
        },
        "required": ["source"],
        "additionalProperties": False,
    }
    _UNSUPPORTED_BINARY_MESSAGE = "Cannot read binary or unsupported file as text. Use inspect for supported metadata; this file format is not readable by the available tools."

    def run(self, arguments: dict[str, Any], workspace: Path) -> ToolResult:
        source = arguments.get("source")
        if not isinstance(source, str) or not source.strip():
            return ToolResult(False, "'source' must be a non-empty string", "BAD_ARGUMENTS")
        max_chars = max(500, min(int(arguments.get("max_chars", 12000)), 30000))
        if urlparse(source).scheme in {"http", "https"}:
            response, error = _get_public_url(source)
            if response is None:
                code = "PRIVATE_URL_BLOCKED" if error and "private" in error.casefold() else "HTTP_ERROR"
                return ToolResult(False, error or "HTTP read failed", code)
            ctype = response.headers.get("content-type", "")
            text = response.text if "html" not in ctype else _strip_html(response.text)
            return ToolResult(True, text[:max_chars])

        path = (workspace / source).resolve() if not Path(source).is_absolute() else Path(source).resolve()
        try:
            path.relative_to(workspace.resolve())
        except ValueError:
            return ToolResult(False, "File must be inside the workspace", "PATH_OUTSIDE_WORKSPACE")
        if not path.exists() or not path.is_file():
            return ToolResult(False, f"File not found: {source}", "NOT_FOUND")
        suffix = path.suffix.lower()
        detected_type = detect_content_type(path)
        try:
            if detected_type == "pdf":
                from pypdf import PdfReader
                text = "\n\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
            elif detected_type == "xlsx":
                import openpyxl
                wb = openpyxl.load_workbook(path.open("rb"), read_only=True, data_only=True)
                parts = []
                for ws in wb.worksheets:
                    parts.append(f"# Sheet: {ws.title}")
                    for row in ws.iter_rows(values_only=True):
                        parts.append("\t".join("" if v is None else str(v) for v in row))
                text = "\n".join(parts)
            elif detected_type == "json":
                text = json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2)
            else:
                data = path.read_bytes()
                if data.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08", b"%PDF-")):
                    return ToolResult(False, self._UNSUPPORTED_BINARY_MESSAGE, "UNSUPPORTED_BINARY")
                try:
                    text = data.decode("utf-8")
                except UnicodeDecodeError:
                    return ToolResult(False, self._UNSUPPORTED_BINARY_MESSAGE, "UNSUPPORTED_BINARY")
                sample = text[:8192]
                if "\x00" in sample:
                    return ToolResult(False, self._UNSUPPORTED_BINARY_MESSAGE, "UNSUPPORTED_BINARY")
                if sample:
                    controls = sum(1 for char in sample if ord(char) < 32 and char not in "\t\r\n")
                    if controls / len(sample) > 0.05:
                        return ToolResult(False, self._UNSUPPORTED_BINARY_MESSAGE, "UNSUPPORTED_BINARY")
        except ImportError as exc:
            return ToolResult(False, f"Missing file-reader dependency: {exc}. Install -e '.[files]'", "MISSING_DEPENDENCY")
        except Exception as exc:
            return ToolResult(False, f"Read failed: {exc}", "READ_ERROR")
        return ToolResult(True, text[:max_chars])
