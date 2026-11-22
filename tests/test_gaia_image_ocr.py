import subprocess
from gaia_small_agent.tools.read import ReadTool
from gaia_small_agent.tools import default_tools
from scripts.run_gaia_eval_shard import build_parser


def test_ocr_is_opt_in_and_default_tools_unchanged(tmp_path, monkeypatch):
    path = tmp_path / "image.png"
    path.write_bytes(b"\\x89PNG\\r\\n\\x1a\\n" + b"fake-png")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="A readable clue", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    base = ReadTool().run({"source": "image.png"}, tmp_path)
    assert not base.ok and base.error_code == "UNSUPPORTED_BINARY"
    assert not calls
    with_ocr = ReadTool(image_ocr_enabled=True).run({"source": "image.png"}, tmp_path)
    assert with_ocr.ok and "readable clue" in with_ocr.content
    assert calls[0][0] == "tesseract"
    assert [t.name for t in default_tools()] == ["search", "read", "inspect", "python"]
    assert not default_tools()[1].image_ocr_enabled


def test_ocr_has_bounded_failure_and_cannot_escape_workspace(tmp_path, monkeypatch):
    (tmp_path / "x.png").write_bytes(b"\\x89PNG\\r\\n\\x1a\\n" + b"fake")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k:
                        subprocess.CompletedProcess(a[0], 0, stdout="", stderr=""))
    assert ReadTool(image_ocr_enabled=True).run({"source":"x.png"},tmp_path).error_code == "OCR_NO_TEXT"
    assert ReadTool(image_ocr_enabled=True).run({"source":"../outside.png"},tmp_path).error_code == "PATH_OUTSIDE_WORKSPACE"


def test_ocr_eval_flags_default_off():
    parser = build_parser()
    argv=["--hf-token","test","--shard-index","0","--work-root","x","--output","x/result.json"]
    assert not parser.parse_args(argv).image_ocr
    assert not parser.parse_args(argv).only_image_attachments
    assert parser.parse_args(argv+["--image-ocr","--only-image-attachments"]).image_ocr
