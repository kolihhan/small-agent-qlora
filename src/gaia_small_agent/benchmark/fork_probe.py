"""Same-prefix tool-decision counterfactual for *non-GAIA* verified tasks.

Replays the SAME frozen assistant tool-call prefix in two isolated, identical
workspaces. Both arms execute tools and hash the *delivered* observations.
Then one arm must answer on the next call without tools; the other can continue
using the unchanged AgentRuntime and budget. No labels or publisher answers
are read here, no training, and no new scoring claims.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable

from ..agent.loop import AgentRuntime
from ..agent.types import AssistantTurn, TraceEvent


def _hash(obj: object) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(raw.encode("utf-8")).hexdigest()


def _workspace_digest(path: Path) -> str:
    if not path.is_dir():
        raise ValueError("both workspaces must exist before any model call")
    records = []
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        if file.is_symlink():
            raise ValueError("symlink workspace contents are not reproducible")
        records.append([str(file.relative_to(path)).replace("\\", "/"),
                        sha256(file.read_bytes()).hexdigest()])
    return _hash(records)


class _FrozenPrefixModel:
    def __init__(self, base, prefix: list[AssistantTurn], *, answer_now: bool):
        self.base = base
        self.prefix = deepcopy(prefix)
        self.answer_now = answer_now
        self.index = 0
        self.unexpected_tool_at_final = False

    def _replay(self):
        if self.index >= len(self.prefix):
            return None
        replay = deepcopy(self.prefix[self.index])
        self.index += 1
        return replay

    def _answer_only(self, messages):
        # Same underlying pinned base model; no special helper or stronger teacher.
        final_fn = getattr(self.base, "complete_final", None)
        if callable(final_fn):
            turn = final_fn(messages, [])
        else:
            turn = self.base.complete(messages, [])
        if turn.tool_calls:
            self.unexpected_tool_at_final = True
            # Never execute a tool from the answer-now branch. Empty final
            # counts as a failure rather than inventing an answer.
            return AssistantTurn(content="")
        return turn

    def complete(self, messages, tools):
        replay = self._replay()
        if replay is not None:
            return replay
        if self.answer_now:
            return self._answer_only(messages)
        return self.base.complete(messages, tools)

    def complete_final(self, messages, tools):
        replay = self._replay()
        if replay is not None:
            return replay
        return self._answer_only(messages)


def _prefix_digest(events: list[TraceEvent], prefix_length: int) -> str:
    payload = [event.to_dict() for event in events
               if event.step <= prefix_length
               and event.kind in {"tool_call", "tool_result"}]
    if not payload or not any(event["kind"] == "tool_result" for event in payload):
        raise ValueError("no delivered tool observation before fork")
    return _hash(payload)


def run_pair(
    question: str,
    *,
    source_id: str,
    prefix: list[AssistantTurn],
    model_factory: Callable,
    tools_factory: Callable,
    answer_workspace: str | Path,
    continue_workspace: str | Path,
    max_steps: int = 12,
) -> dict:
    """Return paired *unscored* outcomes, never raw protected task text.

    Source and tool data must be authorized, reproducible, and non-GAIA.
    Each model_factory call must instantiate the same pinned model/settings.
    The caller separately scores both branches against verified source answers.
    """
    if not source_id or not isinstance(source_id, str):
        raise ValueError("require frozen non-GAIA source identifier")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("missing question")
    if not prefix or len(prefix) >= max_steps:
        raise ValueError("fork needs nonempty tool prefix and remaining steps")
    if any(not isinstance(t, AssistantTurn) or not t.tool_calls for t in prefix):
        raise ValueError("prefix consists only of original verified tool-call turns")
    wa, wc = Path(answer_workspace), Path(continue_workspace)
    if wa.resolve() == wc.resolve():
        raise ValueError("fork workspaces must be distinct")
    snapshot = _workspace_digest(wa)
    if snapshot != _workspace_digest(wc):
        raise ValueError("two fork workspaces do not have identical source fixtures")
    n = len(prefix)
    results = {}
    replay_hashes = {}
    for name, force_final, workspace in (
        ("answer_now", True, wa),
        ("continue_tools", False, wc),
    ):
        model = _FrozenPrefixModel(model_factory(), prefix, answer_now=force_final)
        result = AgentRuntime(model, tools_factory(), max_steps=max_steps).run(
            question, workspace)
        fingerprint = _prefix_digest(result.trace, n)
        replay_hashes[name] = fingerprint
        if name == "continue_tools" and fingerprint != replay_hashes["answer_now"]:
            raise ValueError("pre-fork observations diverged; abort paired comparison")
        later_calls = sum(e.kind == "tool_call" and e.step > n for e in result.trace)
        if force_final and later_calls:
            raise RuntimeError("answer-now arm executed a post-fork tool")
        results[name] = {
            "answer": result.answer, "completed": result.completed,
            "stop_reason": result.stop_reason, "steps": result.metrics.steps,
            "postfork_tool_calls": later_calls,
            "unexpected_tool_at_final": model.unexpected_tool_at_final,
        }
    return {
        "protocol": "same-prefix-non-gaia-fork-v1",
        "source_id": source_id, "source_snapshot_sha256": snapshot,
        "prefix_tool_observation_sha256": replay_hashes["answer_now"],
        "prefix_tool_turns": n, "max_steps": max_steps,
        "no_gold_or_answers_used_to_select_action": True,
        "not_an_accuracy_result": True, "arms": results,
    }
