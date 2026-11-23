"""Bounded, source-anchored *observations*, not inferred true relations.

The state sees only tool observations already delivered to the agent; never task
labels, human decompositions or unseen document content. Cross-document entities
are lexical *candidates*, not semantic entailments. No additional LLM calls.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


_ENTITIES = re.compile(r"\b[A-Z][a-zA-ZÀ-ÿ]{2,}(?:[ -]+[A-Z][a-zA-ZÀ-ÿ]{2,}){0,2}\b")
_STOP = frozenset({
    "The", "This", "That", "When", "Where", "Which", "What", "Who",
    "His", "Her", "Their", "Was", "Were", "And", "For", "With", "From",
    "Text", "Tool", "Source", "Page", "Content", "Question", "Answer",
    "None", "Passages", "Index", "Wikipedia", "Unknown", "File",
})


def _entities(text: str) -> set[str]:
    return {m.group(0).casefold() for m in _ENTITIES.finditer(text)
            if m.group(0) not in _STOP and len(m.group(0)) <= 64}


def _quote(observation: str, question: str, max_chars: int = 95) -> str:
    # Quotes originate ONLY from the portion of the tool result visible to the model.
    # Prefer lines with lexical overlap to the question but do not assert correctness.
    query_words = {w.casefold() for w in re.findall(r"[A-Za-zÀ-ÿ]{4,}", question)}
    snippets = []
    for line in observation.splitlines():
        line = " ".join(line.split())
        if not line or line.startswith(("ToolMeta:", "ToolError[")):
            continue
        if line.startswith("...[truncated"):
            continue
        hits = sum(word in line.casefold() for word in query_words)
        snippets.append((hits, line))
    if not snippets:
        return ""
    ranked = sorted(enumerate(snippets), key=lambda item: (-item[1][0], item[0]))
    chosen = ranked[0][1][1]
    return chosen[:max_chars].replace('"', "'").replace("[", "(").replace("]", ")")


@dataclass(frozen=True)
class _Anchor:
    ref: str
    source: str
    offset: int
    quote: str
    entities: frozenset[str]


class EvidenceState:
    """Keep short direct excerpts and possible cross-source entity overlaps."""

    def __init__(self, question: str, *, max_sources: int = 3, max_chars: int = 350):
        self.question = question
        self.max_sources = max_sources
        self.max_chars = max_chars
        self._anchors: list[_Anchor] = []
        self.reads_seen = 0
        self.bridge_candidates = 0

    def observe(self, tool: str, arguments: dict, *, ok: bool, observation: str) -> None:
        if tool != "read" or not ok:
            return
        source = arguments.get("source")
        if not isinstance(source, str) or not source.strip():
            return
        # Only record successful calls. Runtime already checked path/network access.
        source = source.strip()
        self.reads_seen += 1
        offset = arguments.get("offset_chars", 0)
        if type(offset) is not int or offset < 0:
            offset = 0
        quote = _quote(observation, self.question)
        if not quote:
            return
        # The reference is to a real successful read operation, not to a model claim.
        ref = f"S{self.reads_seen}"
        entity_set = frozenset(_entities(observation[:1500]))
        self._anchors.append(_Anchor(ref, source, offset, quote, entity_set))
        self._anchors = self._anchors[-self.max_sources:]
        overlaps = [a.entities & b.entities
                    for i,a in enumerate(self._anchors)
                    for b in self._anchors[i+1:]
                    if a.source != b.source]
        self.bridge_candidates = len(set().union(*overlaps)) if overlaps else 0

    def summary(self) -> str:
        if not self._anchors:
            return ""
        parts = ["\n[Evidence index: excerpts actually READ; entities are unverified links]"]
        # Prefer most recent anchors, but preserve actual citation identity.
        for a in self._anchors[-2:]:
            # Filenames can contain unusual characters. No new model instruction from it.
            basename = Path(urlsplit(a.source).path).name or a.source
            safe_name = re.sub(r"[\[\]\r\n<>]", "", basename)[:46]
            parts.append(f"[{a.ref} {safe_name}@{a.offset}] \"{a.quote[:80]}\"")
        if len(self._anchors) >= 2:
            shared = set()
            for i,a in enumerate(self._anchors):
                for b in self._anchors[i+1:]:
                    if a.source != b.source:
                        shared |= a.entities & b.entities
            if shared:
                parts.append("Possible shared names (NOT proven relations): " +
                             ", ".join(sorted(shared)[:3]))
            else:
                parts.append("No verified cross-source relation; inspect before concluding.")
        return ("\n".join(parts))[:self.max_chars]

    def safe_metrics(self) -> dict[str, int]:
        return {
            "successful_reads_seen": self.reads_seen,
            "anchors_kept": len(self._anchors),
            "lexical_bridge_candidates": self.bridge_candidates,
        }
