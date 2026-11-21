"""Tests for the bounded 9B CPU resource gate: never misreport a trained checkpoint."""
from __future__ import annotations

from scripts.cpu_v8_stage70_to100_resource_gate import assess, GIB


def test_stock_public_github_cpu_runner_cannot_fit_original_model_download():
    # GitHub standard public Linux runner advertises 4CPU/16GiB/14GB disk;
    # Qwen3.5-9B source weights are roughly 19.3GB.
    report = assess(ram_bytes=16*GIB,disk_free_bytes=14*GIB,
                    cpu_count=4,base_weights_bytes=19_300_000_000)
    assert report["resource_gate"] == "BLOCKED"
    assert "insufficient_free_disk_to_download_original_model_and_restore_checkpoint" in report["reasons"]
    assert "insufficient_ram_for_safe_nine_billion_parameter_cpu_training_trial" in report["reasons"]
    assert report["training_performed"] is False
    assert report["cross_device_exact_optimizer_continuation_not_proven"] is True


def test_unknown_model_size_fails_closed_not_claim_trainable():
    report = assess(ram_bytes=96*GIB,disk_free_bytes=200*GIB,
                    cpu_count=32,base_weights_bytes=None)
    assert report["resource_gate"] == "BLOCKED"
    assert "original_model_source_weight_bytes_unverified" in report["reasons"]


def test_larger_machine_pass_is_resources_only_not_training():
    report = assess(ram_bytes=64*GIB,disk_free_bytes=80*GIB,
                    cpu_count=8,base_weights_bytes=19_300_000_000)
    assert report["resource_gate"] == "RESOURCE_GATE_PASS_ONLY"
    assert report["training_performed"] is False
    assert report["estimated_cpu_step_duration"].startswith("unknown")


def test_disk_margin_accounts_for_raw_shards_and_checkpoint():
    weights=20*GIB
    report=assess(ram_bytes=64*GIB,disk_free_bytes=weights+5*GIB,
                  cpu_count=8,base_weights_bytes=weights)
    assert report["resource_gate"] == "BLOCKED"
    assert report["minimum_trial_floor"]["free_disk_bytes"]==weights+6*GIB
