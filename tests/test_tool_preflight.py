from gaia_small_agent.tools.base import ToolResult


def test_preflight_smokes_exact_default_surface_without_live_search(monkeypatch, tmp_path):
    from gaia_small_agent import tools

    calls = []

    class FakeTool:
        def __init__(self, name):
            self.name = name

        def run(self, arguments, workspace):
            calls.append((self.name, arguments, workspace))
            if self.name == "search":
                raise AssertionError("evaluation preflight must not depend on a live search response")
            content = "42" if self.name == "python" else "p4-tool-preflight"
            return ToolResult(True, content)

    monkeypatch.setattr(tools, "default_tools", lambda: [FakeTool(name) for name in ("search", "read", "inspect", "python")])
    monkeypatch.setattr(tools, "version", lambda package: "test")

    result = tools.preflight_default_tools(tmp_path)

    assert result["tool_names"] == ["search", "read", "inspect", "python"]
    assert [name for name, _, _ in calls] == ["read", "inspect", "python"]
    assert all(item[2] == tmp_path for item in calls)


def test_preflight_fails_before_inference_when_search_dependency_is_unavailable(monkeypatch, tmp_path):
    import pytest
    from importlib.metadata import PackageNotFoundError
    from gaia_small_agent import tools

    class FakeTool:
        def __init__(self, name):
            self.name = name

        def run(self, arguments, workspace):
            return ToolResult(True, "42" if self.name == "python" else "p4-tool-preflight")

    monkeypatch.setattr(tools, "default_tools", lambda: [FakeTool(name) for name in ("search", "read", "inspect", "python")])

    def fake_version(package):
        if package == "ddgs":
            raise PackageNotFoundError(package)
        return "test"

    monkeypatch.setattr(tools, "version", fake_version)

    with pytest.raises(RuntimeError, match="search preflight failed: MISSING_DEPENDENCY"):
        tools.preflight_default_tools(tmp_path)


def test_readiness_check_does_not_require_live_search(monkeypatch, tmp_path):
    from gaia_small_agent import tools

    calls = []

    class FakeTool:
        def __init__(self, name):
            self.name = name

        def run(self, arguments, workspace):
            calls.append(self.name)
            if self.name == "search":
                raise AssertionError("doctor readiness must not make a live web query")
            content = "42" if self.name == "python" else "p4-tool-preflight"
            return ToolResult(True, content)

    monkeypatch.setattr(tools, "default_tools", lambda: [FakeTool(name) for name in ("search", "read", "inspect", "python")])
    monkeypatch.setattr(tools, "version", lambda package: "test")

    result = tools.check_default_tools(tmp_path, live_search=False)

    assert result["ready"] is True
    assert calls == ["read", "inspect", "python"]
    assert result["checks"]["search"]["ok"] is True
    assert result["checks"]["read"]["ok"] is True
    assert result["checks"]["inspect"]["ok"] is True
    assert result["checks"]["python"]["ok"] is True


def test_inspect_supports_extensionless_xlsx_and_pdf(tmp_path):
    import json
    import openpyxl
    from pypdf import PdfWriter
    from gaia_small_agent.tools.inspect import InspectTool

    xlsx = tmp_path / "book"
    wb = openpyxl.Workbook(); wb.active.append(["x", "y"]); wb.save(xlsx)
    pdf = tmp_path / "paper"
    writer = PdfWriter(); writer.add_blank_page(width=72, height=72)
    with pdf.open("wb") as handle: writer.write(handle)

    x = InspectTool().run({"path": xlsx.name}, tmp_path)
    p = InspectTool().run({"path": pdf.name}, tmp_path)
    assert x.ok and json.loads(x.content)["detected_type"] == "xlsx"
    assert "sheets" in json.loads(x.content)
    assert p.ok and json.loads(p.content)["detected_type"] == "pdf"
    assert json.loads(p.content)["pages"] == 1


def test_inspect_rejects_oversized_file_before_parser_work(tmp_path):
    from gaia_small_agent.tools.inspect import InspectTool
    from gaia_small_agent.tools.limits import MAX_WORKSPACE_FILE_BYTES

    path = tmp_path / "large.csv"
    with path.open("wb") as handle:
        handle.seek(MAX_WORKSPACE_FILE_BYTES)
        handle.write(b"x")

    result = InspectTool().run({"path": path.name}, tmp_path)

    assert result.ok is False
    assert result.error_code == "CONTENT_TOO_LARGE"
