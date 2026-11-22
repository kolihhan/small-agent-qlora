"""Optional GAIA research-only benchmark-answer leakage guard.

This is NOT an answer oracle or decontaminated score. It uses only the task's
search query and tool observations, never GAIA reference answers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from gaia_small_agent.tools.base import ToolResult

_BENCHMARK = re.compile(r"\bgaia\b.{0,70}\b(?:benchmark|evaluation|dataset|task|question|answer|solution)\b|\b(?:benchmark|evaluation|dataset)\b.{0,70}\bgaia\b", re.I | re.S)
_ANSWER_FIELD = re.compile(r"\b(?:ground[_ -]?truth|gold[_ -]?answer|correct[_ -]?answer|reference[_ -]?answer|final[_ -]?answer)\b", re.I)
_TASK_MARKER = re.compile(r"\b(?:task[_ -]?id|gaia[_ -]?task|gaia[_ -]?question)\b", re.I)
_ANSWER_TABLE = re.compile(r"\b(?:question|prompt)\s*[:|].{0,250}\b(?:answer|response|solution)\s*[:|]", re.I | re.S)
_WORD = re.compile(r"[A-Za-z0-9]{3,}")


def _query_overlap(query: str, content: str) -> bool:
    """Require eight sequential query tokens, not just common keyword matches."""
    tokens = _WORD.findall(query.casefold())[:60]
    words = _WORD.findall(content.casefold())
    if len(tokens) < 8 or len(words) < 8:
        return False
    windows = {tuple(words[i:i+8]) for i in range(len(words) - 7)}
    return any(tuple(tokens[i:i+8]) in windows for i in range(len(tokens) - 7))


def leakage_signals(query: str, content: str) -> tuple[str, ...]:
    """Conservative suspect flags. False negatives are possible."""
    if not isinstance(content, str) or not content:
        return ()
    benchmark = bool(_BENCHMARK.search(content))
    if benchmark and (_ANSWER_FIELD.search(content) or _TASK_MARKER.search(content)):
        return ("BENCHMARK_WITH_ANSWER_KEY",)
    if benchmark and _query_overlap(query, content):
        return ("BENCHMARK_WITH_TASK_TEXT",)
    if _TASK_MARKER.search(content) and _ANSWER_FIELD.search(content):
        return ("TASK_ID_WITH_ANSWER_KEY",)
    if benchmark and _ANSWER_TABLE.search(content):
        return ("BENCHMARK_QA_TABLE",)
    return ()


@dataclass
class LeakageGate:
    """AgentRuntime observation hook; does not alter normal runs unless enabled."""
    total_screened: int = 0
    blocked: int = 0

    def __call__(self, name: str, result: ToolResult, query: str) -> ToolResult:
        if not result.ok or name not in {"search", "read"}:
            return result
        self.total_screened += 1
        flagged = leakage_signals(query, result.content)
        if not flagged:
            return result
        self.blocked += 1
        # Do not echo/retain any possibly contaminated text or URLs in model context.
        return ToolResult(
            False,
            "Potential public benchmark solution material suppressed. Find independent primary evidence.",
            "BENCHMARK_CONTENT_BLOCKED",
            metadata={"guard": flagged[0]},
        )
