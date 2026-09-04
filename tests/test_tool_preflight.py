from gaia_small_agent.tools.base import ToolResult


def test_preflight_smokes_exact_default_surface(monkeypatch, tmp_path):
    from gaia_small_agent import tools

    calls = []

    class FakeTool:
        def __init__(self, name):
            self.name = name

        def run(self, arguments, workspace):
            calls.append((self.name, arguments, workspace))
            content = "42" if self.name == "python" else "p4-tool-preflight"
            return ToolResult(True, content)

    monkeypatch.setattr(tools, "default_tools", lambda: [FakeTool(name) for name in ("search", "read", "inspect", "python")])
    monkeypatch.setattr(tools, "version", lambda package: "test")

    result = tools.preflight_default_tools(tmp_path)

    assert result["tool_names"] == ["search", "read", "inspect", "python"]
    assert [name for name, _, _ in calls] == result["tool_names"]
    assert all(item[2] == tmp_path for item in calls)


def test_preflight_fails_before_inference_when_a_tool_is_unavailable(monkeypatch, tmp_path):
    import pytest
    from gaia_small_agent import tools

    class FakeTool:
        def __init__(self, name):
            self.name = name

        def run(self, arguments, workspace):
            if self.name == "search":
                return ToolResult(False, "ddgs missing", "MISSING_DEPENDENCY")
            return ToolResult(True, "42" if self.name == "python" else "ok")

    monkeypatch.setattr(tools, "default_tools", lambda: [FakeTool(name) for name in ("search", "read", "inspect", "python")])

    with pytest.raises(RuntimeError, match="search preflight failed: MISSING_DEPENDENCY"):
        tools.preflight_default_tools(tmp_path)


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
