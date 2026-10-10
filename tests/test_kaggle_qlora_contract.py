from pathlib import Path


def test_kaggle_launcher_uses_gpu_and_downloads_results():
    workflow = Path('.github/workflows/kaggle-qlora.yml').read_text(encoding='utf-8')
    assert 'KAGGLE_API_TOKEN' in workflow
    assert 'KAGGLE_USERNAME' in workflow
    assert 'NvidiaTeslaT4' in workflow
    assert 'kaggle kernels push' in workflow
    assert 'kaggle kernels output' in workflow
    assert 'scripts/kaggle_qlora_v1.py' in workflow


def test_kaggle_script_runs_matched_transformers_base_and_adapter():
    script = Path('scripts/kaggle_qlora_v1.py').read_text(encoding='utf-8')
    assert 'Qwen/Qwen3.5-4B' in script
    assert '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a' in script
    assert 'flash-linear-attention' in script
    assert 'causal-conv1d' in script
    assert '--backend' in script and 'transformers' in script
    assert '--adapter' in script
    assert 'compare-policy-evals' in script
