"""PRO frozen generic final answer field extractor, with strict false-positive guards.

Experiment only. Never called with a question, gold reference or tool trace.
No model calls, no inference retraining, no task-domain-dependent prompts.
Do not enable in AgentRuntime until an independent source-shift gate passes.
"""
from __future__ import annotations

import re

EXPLICIT = re.compile(r"(?i)^(?:final\s+)?answer\s*:\s*(.+?)\s*$")
EXCLUDE_START = re.compile(
    r"(?i)^(?:because|therefore|however|hence|in conclusion|"
    r"i (?:cannot|don't|do not)|the answer (?:is|depends))\b"
)
MAX_CHARS = 90
MAX_WORDS = 12


def _short_field(candidate: str) -> str | None:
    candidate = candidate.strip().strip(' "\u201c\u201d' + chr(96))
    if not (1 <= len(candidate) <= MAX_CHARS and
            1 <= len(candidate.split()) <= MAX_WORDS):
        return None
    if EXCLUDE_START.search(candidate):
        return None
    if "\n" in candidate or re.search(r"[!?](?:\s|$)", candidate):
        return None
    # A terminal full stop is not necessarily an error (Dr., dates), but
    # removing punctuation from model's answer should not invent content.
    return candidate


def render_final_answer(raw: str) -> tuple[str, str]:
    """Only lift an explicit final answer field or separated last short line."""
    if not isinstance(raw, str):
        raise TypeError("raw final must be text")
    text = raw.strip()
    if not text or "\n" not in text:
        return text, "unchanged_single_line"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return text, "unchanged"
    last = lines[-1]
    match = EXPLICIT.fullmatch(last)
    if match:
        candidate = _short_field(match.group(1))
        if candidate:
            return candidate, "explicit_final_field"
        return text, "ambiguous_explicit_field"
    # Require final paragraph to be precisely one short standalone answer line
    # after a completely blank separator, not a sentence in an explanation.
    paras = re.split(r"\n\s*\n", text)
    if len(paras) >= 2 and paras[-1].strip() == last:
        candidate = _short_field(last)
        if candidate and not last.endswith((".", "!", "?", ":", ";")):
            return candidate, "standalone_final_line"
    return text, "no_safe_final_field"
