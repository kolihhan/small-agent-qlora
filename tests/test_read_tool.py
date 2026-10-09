import zipfile

from gaia_small_agent.tools.read import ReadTool
from gaia_small_agent.tools.limits import MAX_REMOTE_BYTES, MAX_WORKSPACE_FILE_BYTES


def test_read_blocks_loopback_url_before_http_request(tmp_path):
    result = ReadTool().run({"source": "http://127.0.0.1:8080/secret"}, tmp_path)
    assert result.ok is False
    assert result.error_code == "PRIVATE_URL_BLOCKED"


def test_read_rejects_oversized_workspace_file_before_parsing(tmp_path):
    path = tmp_path / "large.txt"
    with path.open("wb") as handle:
        handle.seek(MAX_WORKSPACE_FILE_BYTES)
        handle.write(b"x")

    result = ReadTool().run({"source": path.name}, tmp_path)

    assert result.ok is False
    assert result.error_code == "CONTENT_TOO_LARGE"
    assert "limit" in result.content.casefold()


def test_read_streams_remote_content_and_rejects_over_budget_body(monkeypatch, tmp_path):
    import gaia_small_agent.tools.read as read_module

    class FakeResponse:
        headers = {"content-type": "text/plain; charset=utf-8"}
        is_redirect = False
        is_permanent_redirect = False

        def __init__(self):
            self.closed = False

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size=65536):
            yield b"x" * MAX_REMOTE_BYTES
            yield b"y"

        @property
        def text(self):
            raise AssertionError("streamed HTTP reader must not access response.text")

        def close(self):
            self.closed = True

    response = FakeResponse()
    monkeypatch.setattr(read_module, "_public_url_error", lambda url: None)
    monkeypatch.setattr(read_module.requests, "get", lambda *args, **kwargs: response)

    result = ReadTool().run({"source": "https://example.com/large"}, tmp_path)

    assert result.ok is False
    assert result.error_code == "CONTENT_TOO_LARGE"
    assert response.closed is True


def test_read_rejects_extensionless_zip_as_unsupported_binary(tmp_path):
    path = tmp_path / "presentation"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("ppt/presentation.xml", "<presentation/>")

    result = ReadTool().run({"source": path.name}, tmp_path)

    assert result.ok is False
    assert result.error_code == "UNSUPPORTED_BINARY"
    assert result.content == "Cannot read binary or unsupported file as text. Use inspect for supported metadata; this file format is not readable by the available tools."


def test_read_rejects_extensionless_invalid_utf8(tmp_path):
    path = tmp_path / "invalid"
    path.write_bytes(b"prefix\xff\xfe\x80suffix")

    result = ReadTool().run({"source": path.name}, tmp_path)

    assert result.ok is False
    assert result.error_code == "UNSUPPORTED_BINARY"
    assert result.content == "Cannot read binary or unsupported file as text. Use inspect for supported metadata; this file format is not readable by the available tools."


def test_read_rejects_extensionless_control_heavy_data(tmp_path):
    path = tmp_path / "controls"
    path.write_bytes(b"a" * 18 + b"\x01\x02")

    result = ReadTool().run({"source": path.name}, tmp_path)

    assert result.ok is False
    assert result.error_code == "UNSUPPORTED_BINARY"
    assert result.content == "Cannot read binary or unsupported file as text. Use inspect for supported metadata; this file format is not readable by the available tools."


def test_read_accepts_extensionless_text_at_control_density_boundary(tmp_path):
    path = tmp_path / "boundary"
    expected = "a" * 19 + "\x01"
    path.write_bytes(expected.encode("ascii"))

    result = ReadTool().run({"source": path.name}, tmp_path)

    assert result.ok is True
    assert result.content == expected


def test_read_accepts_extensionless_utf8_text(tmp_path):
    path = tmp_path / "notes"
    expected = "Hello, café — こんにちは\n"
    path.write_bytes(expected.encode("utf-8"))

    result = ReadTool().run({"source": path.name}, tmp_path)

    assert result.ok is True
    assert result.content == expected


def test_read_supports_extensionless_xlsx_by_content(tmp_path):
    import openpyxl
    path = tmp_path / "attachment"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Data"
    ws.append(["name", "value"])
    ws.append(["alpha", 42])
    wb.save(path)

    result = ReadTool().run({"source": path.name}, tmp_path)

    assert result.ok is True
    assert "# Sheet: Data" in result.content
    assert "alpha\t42" in result.content


def test_read_supports_extensionless_pdf_by_magic(tmp_path):
    from pypdf import PdfWriter
    path = tmp_path / "attachment-pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with path.open("wb") as handle:
        writer.write(handle)

    result = ReadTool().run({"source": path.name}, tmp_path)
    assert result.ok is True


def test_unsupported_binary_message_does_not_send_agent_to_python_file_io(tmp_path):
    path = tmp_path / "unsupported"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ppt/presentation.xml", "<presentation/>")
    result = ReadTool().run({"source": path.name}, tmp_path)
    assert result.ok is False
    assert "python" not in result.content.casefold()
