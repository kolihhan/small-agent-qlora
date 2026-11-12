"""Static headless Colab GPU training and API authorization contract."""
from __future__ import annotations
import ast
from pathlib import Path


def test_official_colab_action_does_not_allocate_before_auth_and_cleans_up_gpu():
    text = Path(".github/workflows/colab-gpu-train.yml").read_text(encoding="utf-8")
    assert "google-colab-cli==0.8.0" in text
    assert "GOOGLE_APPLICATION_CREDENTIALS=" in text
    assert "cred.refresh(Request())" in text
    assert "COLAB_AUTH_REJECTED" in text
    assert text.index("cred.refresh(Request())") < text.index("colab --auth=adc run")
    assert 'trap cleanup EXIT' in text
    assert 'colab --auth=adc stop -s "$session"' in text
    assert "colab --auth=adc download" in text
    assert "actions/upload-artifact@" in text
    assert "workflow_dispatch:" in text
    assert "COLAB_SLOT_GATE" in text
    assert text.index("COLAB_SLOT_AVAILABLE") < text.index("colab --auth=adc run")
    assert "active_assignments=$active" in text
    assert ".github/COLAB_GPU_START" in text
    assert 'if: always()' in text
    assert "colab drivemount" not in text
    assert "echo ${COLAB_TOKEN}" not in text


def test_noninteractive_remote_script_parses_and_only_claims_success_after_adapter():
    source = Path("scripts/colab_full_v4_remote.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert "Qwen/Qwen3.5-9B" in source
    assert 'assert torch.cuda.is_available()' in source
    assert 'adapter_model.safetensors' in source
    assert 'COLAB_GPU' in source
    assert 'hashlib.sha256' in source
    assert 'google.colab' not in source
    assert "drive.mount(" not in source
    assert 'HF_TOKEN' not in source
    assert 'Authorization' not in source
    assert 'prerequisite_colab_secret' not in source
    assert 'max_steps=400' in source
    assert 'sys.path.insert(0,str(repo / "src"))' in source
    assert source.index('sys.path.insert(0,str(repo / "src"))') < source.index("from gaia_small_agent.training.oracle_policy_v3")
    assert '(repo / "src/gaia_small_agent/__init__.py").is_file()' in source
    assert 'total_trainable_actions' in source
    assert 'removed_intentionally_bad_actions' in source
    assert 'require_full_prompt=True' in source
    assert 'GAIA' in source

