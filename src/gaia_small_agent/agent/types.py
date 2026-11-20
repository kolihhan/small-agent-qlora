from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


class ModelCapacityError(RuntimeError):
    def __init__(self):
        super().__init__()


class ModelRuntimeError(RuntimeError):
    _ALLOWED_STOP_REASONS = frozenset({"model_timeout", "model_unavailable", "model_error"})

    _ALLOWED_DIAGNOSTICS = frozenset({
        "MODEL_TIMEOUT", "MODEL_CONNECTION_ERROR", "MODEL_HTTP_4XX",
        "MODEL_HTTP_5XX", "MODEL_HTTP_OTHER", "MODEL_REQUEST_ERROR",
        "MODEL_INVALID_JSON", "MODEL_INVALID_RESPONSE_OBJECT",
        "MODEL_CHOICES_MALFORMED", "MODEL_MESSAGE_MALFORMED",
        "MODEL_TOOL_CALLS_MALFORMED", "MODEL_TOOL_CALL_MALFORMED",
        "MODEL_FUNCTION_MALFORMED", "MODEL_TOOL_ARGUMENTS_JSON_MALFORMED",
        "MODEL_TOOL_ARGUMENTS_TYPE_MALFORMED", "MODEL_UNCLASSIFIED",
    })

    def __init__(self, stop_reason: str, message: str,
                 diagnostic_code: str = "MODEL_UNCLASSIFIED"):
        if stop_reason not in self._ALLOWED_STOP_REASONS:
            raise ValueError(f"invalid model stop reason: {stop_reason}")
        if diagnostic_code not in self._ALLOWED_DIAGNOSTICS:
            raise ValueError("unrecognized safe model diagnostic category")
        self.stop_reason = stop_reason
        self.message = str(message)
        # Explicit finite category, never raw HTTP body / model output / token.
        self.diagnostic_code = diagnostic_code
        super().__init__(self.message)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class AssistantTurn:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning_content: str = ""


@dataclass
class TraceEvent:
    kind: str
    step: int
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunMetrics:
    steps: int = 0
    tool_calls: int = 0
    tool_successes: int = 0
    tool_errors: int = 0
    duplicate_calls_blocked: int = 0
    invalid_tool_calls: int = 0
    tool_retries: int = 0


@dataclass
class AgentResult:
    answer: str
    completed: bool
    stop_reason: str
    trace: list[TraceEvent]
    metrics: RunMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "completed": self.completed,
            "stop_reason": self.stop_reason,
            "metrics": asdict(self.metrics),
            "trace": [event.to_dict() for event in self.trace],
        }
