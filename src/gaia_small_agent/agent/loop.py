from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from .types import AgentResult, AssistantTurn, ModelCapacityError, ModelRuntimeError, RunMetrics, TraceEvent
from ..tools.base import Tool, ToolResult


class Model(Protocol):
    def complete(self, messages: list[dict], tools: list[dict]) -> AssistantTurn: ...


SYSTEM_PROMPT = """You are Small Agent, a compact autonomous tool-using assistant.
Choose the next action from the available tools based only on the user task and observations.
There is no fixed workflow. Recover from tool errors by changing strategy when useful.
Use tools only when needed. Never invent tool results.
When you have enough evidence, return the final answer concisely.
Do not reveal hidden chain-of-thought; tool calls and short factual status are sufficient.
"""
MAX_TOOL_OBSERVATION_CHARS = 2000
_TOOL_OBSERVATION_TRUNCATION = "\n...[truncated from {count} characters; request a narrower result or use another tool call to inspect only what is needed]"
_RETRYABLE_TOOL_ERRORS = frozenset({
    "SEARCH_TIMEOUT",
    "SEARCH_ERROR",
    "FETCH_ERROR",
    "HTTP_ERROR",
    "READ_ERROR",
    "TIMEOUT",
    "TOOL_EXCEPTION",
})


def _bound_tool_observation(observation: str) -> str:
    if len(observation) <= MAX_TOOL_OBSERVATION_CHARS:
        return observation
    marker = _TOOL_OBSERVATION_TRUNCATION.format(count=len(observation))
    return observation[: MAX_TOOL_OBSERVATION_CHARS - len(marker)] + marker


def _looks_like_textual_tool_call(content: str) -> bool:
    return content.lstrip().startswith("<tool_call>")


class AgentRuntime:
    def __init__(self, model: Model, tools: list[Tool], max_steps: int = 12):
        self.model = model
        self.tools = {tool.name: tool for tool in tools}
        self.max_steps = max_steps

    def run(self, question: str, workspace: str | Path) -> AgentResult:
        workspace = Path(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        trace: list[TraceEvent] = []
        metrics = RunMetrics()
        seen: dict[str, ToolResult] = {}
        retry_counts: dict[str, int] = {}
        tool_definitions = [tool.definition() for tool in self.tools.values()]

        for step in range(1, self.max_steps + 1):
            metrics.steps = step
            final_only = step == self.max_steps
            try:
                if final_only:
                    complete_final = getattr(self.model, "complete_final", None)
                    if callable(complete_final):
                        turn = complete_final(messages, tool_definitions)
                    else:
                        turn = self.model.complete(messages, [])
                else:
                    turn = self.model.complete(messages, tool_definitions)
            except ModelCapacityError:
                trace.append(TraceEvent("model_capacity", step, {}))
                return AgentResult("", False, "model_capacity", trace, metrics)
            except ModelRuntimeError as exc:
                trace.append(TraceEvent("model_error", step, {"stop_reason": exc.stop_reason, "message": exc.message}))
                return AgentResult("", False, exc.stop_reason, trace, metrics)

            if not turn.tool_calls:
                answer = (turn.content or "").strip()
                if not answer:
                    trace.append(TraceEvent("empty_final", step, {}))
                    return AgentResult("", False, "empty_final", trace, metrics)
                if _looks_like_textual_tool_call(answer):
                    trace.append(TraceEvent("invalid_final", step, {"content": answer}))
                    return AgentResult("", False, "invalid_final", trace, metrics)
                trace.append(TraceEvent("final", step, {"answer": answer}))
                return AgentResult(answer, True, "final", trace, metrics)

            if final_only:
                trace.append(TraceEvent("max_steps", step, {"blocked_tool_calls": len(turn.tool_calls)}))
                return AgentResult("", False, "max_steps", trace, metrics)

            assistant_tool_calls = []
            for call in turn.tool_calls:
                assistant_tool_calls.append({
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                })
            messages.append({"role": "assistant", "content": turn.content or "", "tool_calls": assistant_tool_calls})

            for call in turn.tool_calls:
                metrics.tool_calls += 1
                trace.append(TraceEvent("tool_call", step, {"id": call.id, "name": call.name, "arguments": call.arguments}))
                key = json.dumps([call.name, call.arguments], sort_keys=True, ensure_ascii=False, default=str)

                previous = seen.get(key)
                can_retry = (
                    previous is not None
                    and not previous.ok
                    and previous.error_code in _RETRYABLE_TOOL_ERRORS
                    and retry_counts.get(key, 0) < 1
                )
                if previous is not None and not can_retry:
                    metrics.duplicate_calls_blocked += 1
                    result = ToolResult(False, "Exact duplicate tool call blocked; choose a different action.", "DUPLICATE_CALL")
                else:
                    if can_retry:
                        retry_counts[key] = retry_counts.get(key, 0) + 1
                        metrics.tool_retries += 1

                    tool = self.tools.get(call.name)
                    if tool is None:
                        metrics.invalid_tool_calls += 1
                        result = ToolResult(False, f"Unknown tool: {call.name}", "UNKNOWN_TOOL")
                    else:
                        argument_error = tool.validate_arguments(call.arguments)
                        if argument_error is not None:
                            metrics.invalid_tool_calls += 1
                            result = ToolResult(False, argument_error, "BAD_ARGUMENTS")
                        else:
                            try:
                                result = tool.run(call.arguments, workspace)
                            except Exception as exc:  # runtime boundary: tool exceptions become observations
                                result = ToolResult(False, f"{type(exc).__name__}: {exc}", "TOOL_EXCEPTION")
                            if result.ok:
                                metrics.tool_successes += 1
                            else:
                                metrics.tool_errors += 1
                    seen[key] = result

                obs = _bound_tool_observation(result.observation())
                trace.append(TraceEvent("tool_result", step, {
                    "id": call.id,
                    "name": call.name,
                    "ok": result.ok,
                    "error_code": result.error_code,
                    "metadata": result.metadata,
                    "content": obs,
                }))
                messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": obs})

        return AgentResult("", False, "max_steps", trace, metrics)
