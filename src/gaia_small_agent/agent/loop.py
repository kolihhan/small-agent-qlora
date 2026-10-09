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
_STEP_LIMIT_FINALIZATION = (
    "The tool budget is exhausted. Do not request another tool. "
    "Using only the evidence already gathered, return the best final answer now."
)


def _bound_tool_observation(observation: str) -> str:
    if len(observation) <= MAX_TOOL_OBSERVATION_CHARS:
        return observation
    marker = _TOOL_OBSERVATION_TRUNCATION.format(count=len(observation))
    return observation[: MAX_TOOL_OBSERVATION_CHARS - len(marker)] + marker


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
        seen: set[str] = set()

        for step in range(1, self.max_steps + 1):
            metrics.steps = step
            try:
                turn = self.model.complete(messages, [t.definition() for t in self.tools.values()])
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
                trace.append(TraceEvent("final", step, {"answer": answer}))
                return AgentResult(answer, True, "final", trace, metrics)

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

                if key in seen:
                    metrics.duplicate_calls_blocked += 1
                    metrics.tool_errors += 1
                    result = ToolResult(False, "Exact duplicate tool call blocked; choose a different action.", "DUPLICATE_CALL")
                else:
                    seen.add(key)
                    tool = self.tools.get(call.name)
                    if tool is None:
                        metrics.tool_errors += 1
                        result = ToolResult(False, f"Unknown tool: {call.name}", "UNKNOWN_TOOL")
                    else:
                        try:
                            result = tool.run(call.arguments, workspace)
                        except Exception as exc:  # runtime boundary: tool exceptions become observations
                            result = ToolResult(False, f"{type(exc).__name__}: {exc}", "TOOL_EXCEPTION")
                        if result.ok:
                            metrics.tool_successes += 1
                        else:
                            metrics.tool_errors += 1

                obs = _bound_tool_observation(result.observation())
                trace.append(TraceEvent("tool_result", step, {
                    "id": call.id,
                    "name": call.name,
                    "ok": result.ok,
                    "error_code": result.error_code,
                    "content": obs,
                }))
                messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": obs})

        finalization_step = self.max_steps + 1
        messages.append({"role": "user", "content": _STEP_LIMIT_FINALIZATION})
        try:
            turn = self.model.complete(messages, [])
        except ModelCapacityError:
            trace.append(TraceEvent("step_limit_finalization_failed", finalization_step, {"reason": "model_capacity"}))
            return AgentResult("", False, "max_steps", trace, metrics)
        except ModelRuntimeError as exc:
            trace.append(TraceEvent("step_limit_finalization_failed", finalization_step, {
                "reason": exc.stop_reason,
                "message": exc.message,
            }))
            return AgentResult("", False, "max_steps", trace, metrics)

        answer = (turn.content or "").strip()
        if not turn.tool_calls and answer:
            trace.append(TraceEvent("final_after_step_limit", finalization_step, {"answer": answer}))
            return AgentResult(answer, True, "final_after_step_limit", trace, metrics)

        trace.append(TraceEvent("step_limit_finalization_failed", finalization_step, {
            "reason": "tool_call_returned" if turn.tool_calls else "empty_final",
        }))
        return AgentResult("", False, "max_steps", trace, metrics)
