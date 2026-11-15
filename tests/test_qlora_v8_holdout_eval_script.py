import json
from hashlib import sha256

from scripts.qlora_v8_holdout_paired import MODEL, validate_inputs


def _fixture(tmp_path):
    holdout = tmp_path / "holdout.jsonl"
    audit = tmp_path / "audit.json"
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    rows = [
        {
            "task_id": f"holdout-{i}",
            "source": "musique-train" if i < 24 else "hotpotqa-distractor-train",
            "messages": [
                {"role": "user", "content": f"question {i}"},
                {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "read", "arguments": {"source": "doc.txt"}}}]},
                {"role": "tool", "content": "evidence"},
                {"role": "assistant", "content": f"answer {i}"},
            ],
            "training_action_indices": [1, 3],
        }
        for i in range(35)
    ]
    holdout.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    weights = b"fake weights for CLI validation only"
    (adapter / "adapter_model.safetensors").write_bytes(weights)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    manifest = {
        "train_model": MODEL, "train_parents": 304, "train_actions": 798,
        "validation_parents": 35, "validation_actions": 176,
        "no_gaia_training": True, "gaia_evaluated": False,
        "validation_sha256": sha256(holdout.read_bytes()).hexdigest(),
        "train_sha256": "a" * 64,
    }
    audit.write_text(json.dumps(manifest), encoding="utf-8")
    checkpoint = {
        "model": MODEL, "optimizer_steps": 100, "train_actions": 798,
        "holdout_parents_unseen": 35, "gaia_evaluated": False,
        "adapter_initialized": "from-base-not-v7",
        "train_sha256": manifest["train_sha256"],
        "source_sha": "a" * 40,
        "checkpoint_sha256": sha256(weights).hexdigest(),
    }
    (adapter / "v8-verified-report.json").write_text(json.dumps(checkpoint), encoding="utf-8")
    return holdout, audit, adapter


def test_preflight_uses_source_sha_and_physical_checkpoint_before_any_gpu():
    holdout, audit, adapter = _fixture(__import__("pathlib").Path(__import__("tempfile").mkdtemp()))
    report = validate_inputs(holdout, audit, adapter)
    assert len(report["selected"]) == 70
    assert report["checkpoint"]["optimizer_steps"] == 100


def test_preflight_rejects_edited_weights_and_edited_holdout(tmp_path):
    import pytest
    holdout, audit, adapter = _fixture(tmp_path)
    (adapter / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest"):
        validate_inputs(holdout, audit, adapter)
    (adapter / "adapter_model.safetensors").write_bytes(b"fake weights for CLI validation only")
    holdout.write_text(holdout.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        validate_inputs(holdout, audit, adapter)


def test_preflight_refuses_incomplete_training_steps(tmp_path):
    import pytest
    holdout, audit, adapter = _fixture(tmp_path)
    p = adapter / "v8-verified-report.json"
    data = json.loads(p.read_text())
    data["optimizer_steps"] = 0
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="identity"):
        validate_inputs(holdout, audit, adapter)
