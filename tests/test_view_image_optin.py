import base64
from pathlib import Path

import pytest

from gaia_small_agent.tools import default_tools
from gaia_small_agent.tools.view_image import ViewImageTool


PNG = b"\x89PNG\r\n\x1a\n" + b"dummy-png-fixture"
JPG = b"\xff\xd8\xff" + b"dummy-jpeg"


def test_vision_is_opt_in_only():
    assert [t.name for t in default_tools()] == ["search", "read", "inspect", "python"]
    assert [t.name for t in default_tools(view_image_enabled=True)] == [
        "search", "read", "inspect", "python", "view_image"
    ]
    assert ViewImageTool.schema["required"] == ["path"]


def test_local_only_endpoint_guards():
    for url in (
        "https://127.0.0.1:8080",
        "http://localhost:8080",
        "http://10.0.0.1:8080",
        "http://127.0.0.1:8080/bad",
        "http://127.0.0.1:8080?x=1",
        "http://attacker:password@127.0.0.1:8080",
    ):
        with pytest.raises(ValueError):
            ViewImageTool(base_url=url)


def test_image_tool_local_bytes_with_no_gold_label_exposure(tmp_path, monkeypatch):
    (tmp_path / "scene.png").write_bytes(PNG)
    requests_seen = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices":[{"message":{"content":"A red square and the number seven."}}]}

    def fake_post(url, **kwargs):
        requests_seen.append((url,kwargs))
        return FakeResponse()

    monkeypatch.setattr("gaia_small_agent.tools.view_image.requests.post",fake_post)
    r = ViewImageTool().run({"path":"scene.png"},tmp_path)
    assert r.ok
    assert "red square" in r.content
    assert len(requests_seen)==1
    url, kw = requests_seen[0]
    assert url=="http://127.0.0.1:8080/v1/chat/completions"
    content=kw["json"]["messages"][0]["content"]
    assert content[0]["type"]=="text" and "answer to a question" in content[0]["text"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert base64.b64decode(content[1]["image_url"]["url"].split(",",1)[1])==PNG
    assert kw["json"]["max_tokens"]==256
    assert "task_id" not in str(kw["json"])
    assert "expected_answer" not in str(kw["json"])


def test_no_path_escape_binary_gif_or_nonexistent_file(tmp_path):
    outside=tmp_path.parent/"VISION_EXTERNAL.png"
    outside.write_bytes(PNG)
    tool=ViewImageTool()
    assert tool.run({"path":str(outside)},tmp_path).error_code=="PATH_OUTSIDE_WORKSPACE"
    assert tool.run({"path":"../VISION_EXTERNAL.png"},tmp_path).error_code=="PATH_OUTSIDE_WORKSPACE"
    (tmp_path/"link.png").symlink_to(outside)
    assert tool.run({"path":"link.png"},tmp_path).error_code in {"PATH_OUTSIDE_WORKSPACE","NOT_FOUND"}
    assert tool.run({"path":"missing.png"},tmp_path).error_code=="NOT_FOUND"
    (tmp_path/"fake.gif").write_bytes(b"GIF89a")
    assert tool.run({"path":"fake.gif"},tmp_path).error_code=="UNSUPPORTED_IMAGE"


def test_no_private_model_call_for_bad_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr("gaia_small_agent.tools.view_image.requests.post",
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not connect")))
    tool=ViewImageTool()
    assert tool.run({"path":"not-there.png"},tmp_path).error_code=="NOT_FOUND"
    (tmp_path/"oversize.png").write_bytes(PNG + b"0"*(tool.MAX_IMAGE_BYTES + 1))
    assert tool.run({"path":"oversize.png"},tmp_path).error_code=="CONTENT_TOO_LARGE"


def test_vision_returns_bounded_text_and_handles_api_failures(tmp_path, monkeypatch):
    (tmp_path/"s.jpg").write_bytes(JPG)
    class FakeResponse:
        def raise_for_status(self):
            pass
        def json(self):
            return {"choices":[{"message":{"content":"x"*9000}}]}
    monkeypatch.setattr("gaia_small_agent.tools.view_image.requests.post",lambda *a,**k:FakeResponse())
    result=ViewImageTool().run({"path":"s.jpg"},tmp_path)
    assert result.ok and len(result.content)==ViewImageTool.MAX_RESPONSE_CHARS
    monkeypatch.setattr("gaia_small_agent.tools.view_image.requests.post",
                        lambda *a,**k: (_ for _ in ()).throw(ValueError("failure")))
    assert ViewImageTool().run({"path":"s.jpg"},tmp_path).error_code=="VISION_UNAVAILABLE"
