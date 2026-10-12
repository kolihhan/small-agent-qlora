from gaia_small_agent.cli import build_parser, make_runtime


def test_run_parser_supports_openai_compat_backend():
    args = build_parser().parse_args([
        "run",
        "hello",
        "--backend",
        "openai_compat",
        "--model",
        "qwen3.5-9b",
        "--openai-base-url",
        "http://127.0.0.1:8080",
    ])

    assert args.backend == "openai_compat"
    assert args.model == "qwen3.5-9b"
    assert args.openai_base_url == "http://127.0.0.1:8080"


def test_make_runtime_builds_openai_compat_model(monkeypatch):
    import gaia_small_agent.cli as cli

    captured = {}

    class FakeModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(cli, "OpenAICompatModel", FakeModel)
    args = build_parser().parse_args([
        "run",
        "hello",
        "--backend",
        "openai_compat",
        "--model",
        "qwen3.5-9b",
        "--openai-base-url",
        "http://127.0.0.1:18080",
        "--max-new-tokens",
        "640",
    ])

    runtime = make_runtime(args)

    assert isinstance(runtime.model, FakeModel)
    assert captured == {
        "model": "qwen3.5-9b",
        "base_url": "http://127.0.0.1:18080",
        "max_new_tokens": 640,
    }
