"""Opt-in, deterministic evidence locator across files already present in the task workspace.

No web, model, embeddings, training/gold labels, network, or access outside workspace.
The caller still has to read the located sources and reason about the evidence.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .base import Tool, ToolResult
from .limits import MAX_WORKSPACE_FILE_BYTES


class WorkspaceFindTool(Tool):
    name = "workspace_find"
    description = (
        "Find exact words or phrases in available workspace text files. Returns file paths, "
        "line numbers and short context excerpts, not an answer. Use read on the cited file "
        "to check evidence and follow additional links."
    )
    schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string", "minLength": 2, "maxLength": 120,
                "description": "Exact entity name or phrase to locate across workspace documents",
            },
            "max_hits": {"type": "integer", "minimum": 1, "maximum": 8},
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    _TEXT = frozenset({".txt", ".md", ".csv", ".tsv", ".json"})
    _MAX_FILES = 120
    _MAX_BYTES = 256 * 1024
    _MAX_QUERY_LEN = 120

    def run(self, arguments: dict[str, Any], workspace: Path) -> ToolResult:
        query = arguments.get("query")
        max_hits = arguments.get("max_hits", 5)
        if not isinstance(query, str) or len(query.strip()) < 2 or len(query) > self._MAX_QUERY_LEN:
            return ToolResult(False, "Query must be between 2 and 120 characters", "BAD_ARGUMENTS")
        if isinstance(max_hits, bool) or not isinstance(max_hits, int) or not 1 <= max_hits <= 8:
            return ToolResult(False, "max_hits must be between 1 and 8", "BAD_ARGUMENTS")
        query = " ".join(query.split())
        if not query or "\x00" in query:
            return ToolResult(False, "Invalid query", "BAD_ARGUMENTS")
        workspace = workspace.resolve()
        if not workspace.is_dir():
            return ToolResult(False, "Workspace directory unavailable", "NOT_FOUND")
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        hits = []
        scanned = 0
        skipped_oversized = 0
        for path in sorted(workspace.rglob("*")):
            if not path.is_file() or path.is_symlink() or path.suffix.casefold() not in self._TEXT:
                continue
            try:
                resolved = path.resolve(strict=True)
                resolved.relative_to(workspace)
            except (ValueError, OSError):
                continue
            # Never read huge or hidden tool-generated traces, even if accidentally present.
            if any(part.startswith(".") or part in {"runs", "__pycache__"} for part in path.relative_to(workspace).parts):
                continue
            if path.stat().st_size > min(MAX_WORKSPACE_FILE_BYTES, self._MAX_BYTES):
                skipped_oversized += 1
                continue
            scanned += 1
            if scanned > self._MAX_FILES:
                break
            rel = path.relative_to(workspace).as_posix()
            try:
                data = path.read_text(encoding="utf-8")
            except (UnicodeError, OSError):
                continue
            for number, line in enumerate(data.splitlines(), 1):
                found = pattern.search(line)
                if not found:
                    continue
                left = max(0, found.start() - 90)
                right = min(len(line), found.end() + 90)
                excerpt = line[left:right].strip().replace("\t", " ")
                hits.append({
                    "path": rel,
                    "line": number,
                    "excerpt": excerpt,
                })
                if len(hits) >= max_hits:
                    break
            if len(hits) >= max_hits:
                break
        # Deterministic finite output, names and line numbers are sourced from actual files.
        body = "\n".join(
            f"{x['path']}:{x['line']}: {x['excerpt']}" for x in hits
        ) if hits else "No matching text in workspace files; try another entity phrase."
        return ToolResult(True, body, metadata={
            "hits": len(hits),
            "files_scanned": min(scanned, self._MAX_FILES),
            "files_skipped_oversized": skipped_oversized,
            "search_scope": "workspace_only",
        })
