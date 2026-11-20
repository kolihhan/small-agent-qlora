"""Opt-in verbatim final answer extraction using the SAME original model.

Frozen from the 2026-11-17 public Hotpot/2Wiki experiments. This post-processes
only completed answers; it does not alter tools or reasoning. Fall back to
the original answer whenever the extraction cannot be verified.
"""
from __future__ import annotations

import json
import re
from typing import Protocol

from .types import AssistantTurn, ModelRuntimeError


class TextModel(Protocol):
    def complete(self, messages: list[dict], tools: list[dict]) -> AssistantTurn: ...


SERIALIZER_RULE=(
    "You are a lossless answer-field serializer. You receive a question and an "
    "earlier assistant reply. You MUST NOT answer the question from memory, "
    "browse, reason with outside facts, or change the earlier reply's answer. "
    "Extract the shortest single answer span that the EARLIER REPLY explicitly "
    "asserts as the answer to the question: a proper name, number, year, "
    "place, yes/no, or short phrase. Preserve spelling and meaning. "
    "Do not select a distractor or an answer the reply rejected. "
    "If the reply is uncertain, conflicting, missing the answer, or cannot "
    "be reduced safely, output the earlier reply UNCHANGED. "
    "Do not include any explanation, quotes, prefixes, markdown or extra text."
)


def serialize_answer(question: str, answer: str, model: TextModel) -> tuple[str, str]:
    """One same-model call, then a strictly bounded substring check."""
    if not answer.strip():
        return answer, "empty_original"
    user = json.dumps({"question": question, "earlier_reply": answer},
                      ensure_ascii=False, sort_keys=True)
    try:
        turn = model.complete([
            {"role": "system", "content": SERIALIZER_RULE},
            {"role": "user", "content": user},
        ], tools=[])
    except ModelRuntimeError:
        return answer, "model_error"
    candidate = turn.content.strip()
    if (turn.tool_calls or not candidate or len(candidate) > 96
        or "\n" in candidate
        or not re.search(re.escape(candidate), answer, flags=re.IGNORECASE)):
        return answer, "rejected_not_verbatim_or_not_short"
    if re.search(r"(?i)\b(therefore|because|however|uncertain|not enough information)\b",
                 candidate):
        return answer, "rejected_nonanswer"
    return candidate, "accepted_verbatim"
