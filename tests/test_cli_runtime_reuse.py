from types import SimpleNamespace


def test_shared_runtime_factory_loads_model_once(monkeypatch):
    import gaia_small_agent.cli as cli

    calls = []
    sentinel = object()

    def fake_make_runtime(args):
        calls.append(args)
        return sentinel

    monkeypatch.setattr(cli, "make_runtime", fake_make_runtime)
    args = SimpleNamespace()
    factory = cli._shared_runtime_factory(args)

    assert factory() is sentinel
    assert factory() is sentinel
    assert len(calls) == 1
