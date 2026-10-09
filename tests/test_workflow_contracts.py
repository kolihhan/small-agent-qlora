from pathlib import Path


def test_gaia_shadow_workflow_is_matched_and_aggregate_only():
    path = Path(".github/workflows/gaia-shadow-gate.yml")
    assert path.exists()
    text = path.read_text(encoding="utf-8")

    assert "HF_TOKEN_NOT_CONFIGURED" in text
    assert "baseline_sha=" in Path(".github/GAIA_SHADOW_TRIGGER").read_text(encoding="utf-8")
    assert "exposure=" in Path(".github/GAIA_SHADOW_TRIGGER").read_text(encoding="utf-8")
    assert "candidate_name=" in Path(".github/GAIA_SHADOW_TRIGGER").read_text(encoding="utf-8")

    assert "ref: ${{ env.BASELINE_SHA }}" in text
    assert "path: baseline" in text
    assert text.count("--partition shadow") == 2
    assert text.count("--max-steps 12") == 2
    assert text.count("--max-new-tokens 512") == 2
    assert "qwen3.5:4b" in text
    assert "evaluate_shadow_promotion" in text
    assert "gaia-shadow-summary.json" in text

    upload_block = text.split("Upload sanitized aggregate", 1)[1]
    assert "gaia-shadow-summary.json" in upload_block
    for forbidden in (
        "results.jsonl",
        "manifest.json",
        "protected-question",
        "trace",
        "gaia-baseline",
        "gaia-candidate",
    ):
        assert forbidden not in upload_block


def test_gaia_shadow_workflow_validates_exposure_budget():
    text = Path(".github/workflows/gaia-shadow-gate.yml").read_text(encoding="utf-8")

    assert "EXPOSURE" in text
    assert '"$EXPOSURE" -lt 1' in text
    assert '"$EXPOSURE" -gt 3' in text
