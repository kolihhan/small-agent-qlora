import requests

from gaia_small_agent.doctor import run_doctor


def _tool_report(*, ready=True):
    return {
        "ready": ready,
        "tool_names": ["search", "read", "inspect", "python"],
        "dependencies": {"requests": "test", "ddgs": "test", "pypdf": "test", "openpyxl": "test"},
        "checks": {
            "search": {"ok": True, "detail": "dependency available"},
            "read": {"ok": ready, "detail": "ok" if ready else "read failed"},
            "inspect": {"ok": True, "detail": "ok"},
            "python": {"ok": True, "detail": "ok"},
        },
        "smokes": {},
    }


def test_ollama_doctor_reports_ready_when_backend_model_and_tools_are_ready(monkeypatch, tmp_path):
    import gaia_small_agent.doctor as doctor

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"models": [{"name": "qwen3.5:4b"}]}

    monkeypatch.setattr(doctor, "check_default_tools", lambda workspace, live_search=False: _tool_report())
    monkeypatch.setattr(doctor.requests, "get", lambda *args, **kwargs: FakeResponse())

    report = run_doctor("ollama", "qwen3.5:4b", "http://localhost:11434", tmp_path)

    assert report["ready"] is True
    assert report["checks"]["backend"]["ok"] is True
    assert report["checks"]["model"]["ok"] is True
    assert report["checks"]["tools"]["ok"] is True
    assert report["tools"]["tool_names"] == ["search", "read", "inspect", "python"]


def test_ollama_doctor_reports_unavailable_backend_without_hiding_tool_report(monkeypatch, tmp_path):
    import gaia_small_agent.doctor as doctor

    monkeypatch.setattr(doctor, "check_default_tools", lambda workspace, live_search=False: _tool_report())

    def fail_get(*args, **kwargs):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(doctor.requests, "get", fail_get)

    report = run_doctor("ollama", "qwen3.5:4b", "http://localhost:11434", tmp_path)

    assert report["ready"] is False
    assert report["checks"]["backend"]["ok"] is False
    assert report["checks"]["model"]["ok"] is False
    assert report["checks"]["tools"]["ok"] is True
    assert "refused" in report["checks"]["backend"]["detail"]


def test_ollama_doctor_reports_missing_requested_model(monkeypatch, tmp_path):
    import gaia_small_agent.doctor as doctor

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"models": [{"name": "other:latest"}]}

    monkeypatch.setattr(doctor, "check_default_tools", lambda workspace, live_search=False: _tool_report())
    monkeypatch.setattr(doctor.requests, "get", lambda *args, **kwargs: FakeResponse())

    report = run_doctor("ollama", "qwen3.5:4b", "http://localhost:11434", tmp_path)

    assert report["ready"] is False
    assert report["checks"]["backend"]["ok"] is True
    assert report["checks"]["model"]["ok"] is False
    assert "qwen3.5:4b" in report["checks"]["model"]["detail"]


def test_doctor_aggregates_tool_failure_instead_of_aborting(monkeypatch, tmp_path):
    import gaia_small_agent.doctor as doctor

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"models": [{"name": "qwen3.5:4b"}]}

    tool_report = _tool_report(ready=False)
    monkeypatch.setattr(doctor, "check_default_tools", lambda workspace, live_search=False: tool_report)
    monkeypatch.setattr(doctor.requests, "get", lambda *args, **kwargs: FakeResponse())

    report = run_doctor("ollama", "qwen3.5:4b", "http://localhost:11434", tmp_path)

    assert report["ready"] is False
    assert report["checks"]["backend"]["ok"] is True
    assert report["checks"]["model"]["ok"] is True
    assert report["checks"]["tools"]["ok"] is False
    assert report["tools"]["checks"]["read"]["ok"] is False
    assert report["tools"]["checks"]["inspect"]["ok"] is True


def test_transformers_doctor_checks_dependencies_without_loading_model_or_calling_ollama(monkeypatch, tmp_path):
    import gaia_small_agent.doctor as doctor

    required = {"torch", "transformers", "bitsandbytes", "accelerate"}
    seen = []

    def fake_find_spec(name):
        seen.append(name)
        return object() if name in required else None

    monkeypatch.setattr(doctor, "check_default_tools", lambda workspace, live_search=False: _tool_report())
    monkeypatch.setattr(doctor, "find_spec", fake_find_spec)
    monkeypatch.setattr(doctor.requests, "get", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("transformers doctor must not call Ollama")))

    report = run_doctor("transformers", "Qwen/Qwen3.5-4B", "http://localhost:11434", tmp_path)

    assert report["ready"] is True
    assert set(seen) == required
    assert report["checks"]["backend"]["ok"] is True
    assert report["checks"]["model"]["ok"] is True
    assert "not loaded" in report["checks"]["model"]["detail"].lower()
