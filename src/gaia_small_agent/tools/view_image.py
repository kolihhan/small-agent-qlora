"""Opt-in local Qwen vision tool. Requires llama-server with the *matching* mmproj GGUF.

Image bytes never leave loopback and task labels are never added to the vision
request. This is NOT generic OCR or a claim of improved GAIA task accuracy.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from .base import Tool, ToolResult


class ViewImageTool(Tool):
    name = "view_image"
    description = (
        "Visually inspect a local PNG/JPEG/WebP image using the existing local multimodal "
        "Qwen model and return a factual description, text, objects and spatial relations. "
        "Requires the matching mmproj projector; may be uncertain. Use image evidence, not guesses."
    )
    schema = {
        "type": "object",
        "properties": {"path": {
            "type": "string", "minLength": 1,
            "description": "Path relative to the task workspace to a PNG/JPEG/WebP image",
        }},
        "required": ["path"], "additionalProperties": False,
    }
    MAX_IMAGE_BYTES = 8 * 1024 * 1024
    MAX_RESPONSE_CHARS = 1800
    PROMPT = (
        "Describe only what is directly visible in this image. Note the main objects, "
        "visible colors, text and numbers (if legible), spatial relationships, any chart axes/legends, "
        "and uncertainty. Do not invent unreadable text, hidden context, sources, "
        "or an answer to a question not shown here. Be factual and concise."
    )

    def __init__(self, base_url: str = "http://127.0.0.1:8080", timeout_s: float = 120.0):
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/"):
            raise ValueError("Vision inference must use strictly local 127.0.0.1 HTTP server")
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)

    @staticmethod
    def _mime(data: bytes) -> str | None:
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if data.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return "image/webp"
        return None

    def run(self, arguments: dict[str, Any], workspace: Path) -> ToolResult:
        raw = arguments.get("path")
        if not isinstance(raw, str) or not raw.strip():
            return ToolResult(False, "'path' must be a non-empty workspace-relative path", "BAD_ARGUMENTS")
        path = Path(raw.strip())
        if path.is_absolute():
            return ToolResult(False, "Absolute paths are not allowed", "PATH_OUTSIDE_WORKSPACE")
        workspace = workspace.resolve()
        resolved = (workspace / path).resolve()
        try:
            resolved.relative_to(workspace)
        except ValueError:
            return ToolResult(False, "File must stay inside workspace", "PATH_OUTSIDE_WORKSPACE")
        if not resolved.is_file() or resolved.is_symlink() or (workspace / path).is_symlink():
            return ToolResult(False, "Local image file not found or symlink forbidden", "NOT_FOUND")
        if resolved.stat().st_size > self.MAX_IMAGE_BYTES:
            return ToolResult(False, "Image exceeds safe byte limit", "CONTENT_TOO_LARGE")
        data = resolved.read_bytes()
        mime = self._mime(data)
        if mime is None:
            return ToolResult(False, "Image format unsupported: use PNG/JPEG/WebP", "UNSUPPORTED_IMAGE")
        payload = {
            "model": "qwen3.5-9b",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": self.PROMPT},
                    {"type": "image_url", "image_url": {
                        "url": f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
                    }},
                ],
            }],
            "max_tokens": 256,
            "temperature": 0,
            "stream": False,
        }
        try:
            response = requests.post(
                self.base_url + "/v1/chat/completions", json=payload,
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            body = response.json()
        except (requests.RequestException, ValueError) as exc:
            return ToolResult(False, f"Local vision inference unavailable: {type(exc).__name__}", "VISION_UNAVAILABLE")
        try:
            description = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return ToolResult(False, "Vision server returned malformed output", "VISION_BAD_RESPONSE")
        if not isinstance(description, str) or not description.strip():
            return ToolResult(False, "Vision server returned an empty description", "VISION_EMPTY")
        return ToolResult(
            True, description.strip()[:self.MAX_RESPONSE_CHARS],
            metadata={
                "source_path": path.as_posix(),
                "media_type": mime,
                "vision": "local_qwen_with_mmproj",
                "warning": "Model-generated visual description; verify uncertain details",
            }
        )
