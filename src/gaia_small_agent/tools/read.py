from __future__ import annotations

import asyncio
import ipaddress
import json
import re
import socket
from html import unescape
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests

from .base import Tool, ToolResult
from .content_type import detect_content_type
from .limits import MAX_REMOTE_BYTES, MAX_WORKSPACE_FILE_BYTES


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
            item[4][0] for item in socket.getaddrinfo(
                hostname,
                parsed.port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
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
                stream=True,
            )
        except Exception as exc:
            return None, f"HTTP read failed: {exc}"
        if response.is_redirect or response.is_permanent_redirect:
            location = response.headers.get("location")
            response.close()
            if not location:
                return None, "HTTP redirect missing Location header"
            current = urljoin(current, location)
            continue
        try:
            response.raise_for_status()
        except Exception as exc:
            response.close()
            return None, f"HTTP read failed: {exc}"
        return response, None
    return None, f"HTTP redirect limit exceeded ({max_redirects})"


def _read_bounded_response(response, *, max_bytes: int = MAX_REMOTE_BYTES) -> tuple[bytes | None, str | None]:
    raw_length = response.headers.get("content-length")
    if raw_length:
        try:
            if int(raw_length) > max_bytes:
                return None, f"Remote content exceeds {max_bytes} byte limit"
        except ValueError:
            pass

    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        total += len(chunk)
        if total > max_bytes:
            return None, f"Remote content exceeds {max_bytes} byte limit"
        chunks.append(chunk)
    return b"".join(chunks), None


def _extract_pdf_text(reader) -> str:
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def _slice_read_content(text: str, *, offset_chars: int, max_chars: int) -> ToolResult:
    total_chars = len(text)
    start = min(offset_chars, total_chars)
    end = min(start + max_chars, total_chars)
    content = text[start:end]
    return ToolResult(
        True,
        content,
        metadata={
            "offset_chars": start,
            "returned_chars": len(content),
            "total_chars": total_chars,
            "next_offset_chars": end if end < total_chars else None,
        },
    )


class ReadTool(Tool):
    name = "read"
    description = (
        "Read a public URL or a file inside the workspace. Supports common text plus PDF/XLSX when optional "
        "dependencies are installed. Use offset_chars to continue reading long content."
    )
    schema = {
        "type": "object",
        "properties": {
            "source": {"type": "string", "description": "URL or workspace-relative file path"},
            "max_chars": {"type": "integer", "minimum": 500, "maximum": 30000},
            "offset_chars": {
                "type": "integer",
                "minimum": 0,
                "description": "Character offset for continuing a previously truncated read",
            },
        },
        "required": ["source"],
        "additionalProperties": False,
    }
    _UNSUPPORTED_BINARY_MESSAGE = "Cannot read binary or unsupported file as text. Use inspect for supported metadata; this file format is not readable by the available tools."

    def __init__(self, timeout_s: float = 20.0, endpoint: str = "https://search.parallel.ai/mcp"):
        self.timeout_s = float(timeout_s)
        self.endpoint = endpoint

    async def _fetch_parallel(self, url: str) -> ToolResult:
        try:
            from mcp import Client
            from mcp.types import TextContent
        except ImportError:
            return ToolResult(False, "Parallel fetch dependency unavailable", "MISSING_DEPENDENCY")

        try:
            async with Client(self.endpoint) as client:
                result = await client.call_tool(
                    "web_fetch",
                    {"urls": [url], "objective": "Full page content"},
                )
        except Exception as exc:
            return ToolResult(False, f"Parallel fetch failed: {exc}", "FETCH_ERROR")

        text = "\n\n".join(
            block.text for block in result.content if isinstance(block, TextContent)
        ).strip()
        if result.is_error:
            return ToolResult(False, text or "Parallel fetch failed", "FETCH_ERROR")
        if not text:
            return ToolResult(False, "Parallel fetch returned no content", "FETCH_ERROR")

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return ToolResult(True, text)
        if not isinstance(payload, dict):
            return ToolResult(False, "Parallel fetch returned an unexpected payload", "FETCH_ERROR")

        for raw in payload.get("results") or []:
            if not isinstance(raw, dict):
                continue
            content = raw.get("full_content") or raw.get("content")
            if not content:
                excerpts = raw.get("excerpts") or []
                if isinstance(excerpts, str):
                    content = excerpts
                elif isinstance(excerpts, list):
                    content = "\n\n".join(str(item) for item in excerpts if item)
            if content:
                return ToolResult(True, str(content))

        errors = payload.get("errors") or []
        if errors and isinstance(errors[0], dict):
            error = errors[0]
            detail = error.get("content") or error.get("error_type") or "extraction failed"
            return ToolResult(False, str(detail), "FETCH_ERROR")
        return ToolResult(False, "Parallel fetch returned no content", "FETCH_ERROR")

    def run(self, arguments: dict[str, Any], workspace: Path) -> ToolResult:
        source = arguments.get("source")
        if not isinstance(source, str) or not source.strip():
            return ToolResult(False, "'source' must be a non-empty string", "BAD_ARGUMENTS")
        source = source.strip()

        raw_max_chars = arguments.get("max_chars", 12000)
        raw_offset_chars = arguments.get("offset_chars", 0)
        if not isinstance(raw_max_chars, int) or isinstance(raw_max_chars, bool):
            return ToolResult(False, "'max_chars' must be an integer", "BAD_ARGUMENTS")
        if not isinstance(raw_offset_chars, int) or isinstance(raw_offset_chars, bool):
            return ToolResult(False, "'offset_chars' must be an integer", "BAD_ARGUMENTS")
        if raw_offset_chars < 0:
            return ToolResult(False, "'offset_chars' must be >= 0", "BAD_ARGUMENTS")
        max_chars = max(500, min(raw_max_chars, 30000))
        offset_chars = raw_offset_chars

        if urlparse(source).scheme in {"http", "https"}:
            public_error = _public_url_error(source)
            if public_error:
                code = "PRIVATE_URL_BLOCKED" if "private" in public_error.casefold() else "HTTP_ERROR"
                return ToolResult(False, public_error, code)

            try:
                fetched = asyncio.run(
                    asyncio.wait_for(self._fetch_parallel(source), timeout=self.timeout_s)
                )
            except (TimeoutError, RuntimeError):
                fetched = ToolResult(False, "Parallel fetch unavailable", "FETCH_ERROR")
            if fetched.ok:
                return _slice_read_content(fetched.content, offset_chars=offset_chars, max_chars=max_chars)

            response, error = _get_public_url(source, timeout=self.timeout_s)
            if response is None:
                code = "PRIVATE_URL_BLOCKED" if error and "private" in error.casefold() else "HTTP_ERROR"
                return ToolResult(False, error or "HTTP read failed", code)
            try:
                data, size_error = _read_bounded_response(response)
                if size_error:
                    return ToolResult(False, size_error, "CONTENT_TOO_LARGE")
                data = data or b""
                ctype = response.headers.get("content-type", "").casefold()
                if "pdf" in ctype or data.startswith(b"%PDF-"):
                    try:
                        from pypdf import PdfReader
                    except ImportError as exc:
                        return ToolResult(False, f"Missing file-reader dependency: {exc}. Install -e '.[files]'", "MISSING_DEPENDENCY")
                    text = _extract_pdf_text(PdfReader(BytesIO(data)))
                    return _slice_read_content(text, offset_chars=offset_chars, max_chars=max_chars)
                encoding = getattr(response, "encoding", None) or "utf-8"
                text = data.decode(encoding, errors="replace")
                if "html" in ctype:
                    text = _strip_html(text)
                return _slice_read_content(text, offset_chars=offset_chars, max_chars=max_chars)
            except Exception as exc:
                return ToolResult(False, f"HTTP read failed: {exc}", "HTTP_ERROR")
            finally:
                response.close()

        path = (workspace / source).resolve() if not Path(source).is_absolute() else Path(source).resolve()
        try:
            path.relative_to(workspace.resolve())
        except ValueError:
            return ToolResult(False, "File must be inside the workspace", "PATH_OUTSIDE_WORKSPACE")
        if not path.exists() or not path.is_file():
            return ToolResult(False, f"File not found: {source}", "NOT_FOUND")
        if path.stat().st_size > MAX_WORKSPACE_FILE_BYTES:
            return ToolResult(False, f"File exceeds {MAX_WORKSPACE_FILE_BYTES} byte limit", "CONTENT_TOO_LARGE")
        detected_type = detect_content_type(path)
        try:
            if detected_type == "pdf":
                from pypdf import PdfReader
                text = _extract_pdf_text(PdfReader(path))
            elif detected_type == "xlsx":
                import openpyxl
                wb = openpyxl.load_workbook(path.open("rb"), read_only=True, data_only=True)
                try:
                    parts = []
                    for ws in wb.worksheets:
                        parts.append(f"# Sheet: {ws.title}")
                        for row in ws.iter_rows(values_only=True):
                            parts.append("\t".join("" if v is None else str(v) for v in row))
                    text = "\n".join(parts)
                finally:
                    wb.close()
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
        return _slice_read_content(text, offset_chars=offset_chars, max_chars=max_chars)
