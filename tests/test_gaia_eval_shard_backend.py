from scripts import run_gaia_eval_shard as shard_eval


def test_shard_parser_accepts_openai_compat_backend():
    args = shard_eval.build_parser().parse_args([
        "--backend", "openai_compat",
        "--model", "qwen3.5-9b",
        "--openai-base-url", "http://127.0.0.1:8080",
        "--hf-token", "token",
        "--shard-index", "0",
        "--work-root", "runs/test",
        "--output", "runs/test.json",
    ])
    assert args.backend == "openai_compat"
    assert args.openai_base_url == "http://127.0.0.1:8080"


def test_build_model_uses_openai_compat(monkeypatch):
    captured = {}

    class FakeOpenAICompatModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(shard_eval, "OpenAICompatModel", FakeOpenAICompatModel)
    args = shard_eval.build_parser().parse_args([
        "--backend", "openai_compat",
        "--model", "qwen3.5-9b",
        "--openai-base-url", "http://127.0.0.1:18080",
        "--hf-token", "token",
        "--shard-index", "0",
        "--work-root", "runs/test",
        "--output", "runs/test.json",
    ])

    model = shard_eval.build_model(args)

    assert isinstance(model, FakeOpenAICompatModel)
    assert captured == {
        "model": "qwen3.5-9b",
        "base_url": "http://127.0.0.1:18080",
        "max_new_tokens": 512,
    }
