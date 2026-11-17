"""Source/metric contract tests; zero model inference."""
from hashlib import sha256
from research.agent_factorial_gate import audit, _mcnemar, BASE_DIGEST

def cohort():
    tasks = []
    for i in range(24):
        tasks.append({
            "task_sha256": sha256(str(i).encode()).hexdigest(),
            "family": "source_a" if i < 12 else "source_b",
            "arms": {
                "base_native": {"correct": i < 4, "model_calls": 2, "elapsed_s": 1},
                "base_serializer": {"correct": i < 6, "model_calls": 3, "elapsed_s": 1.5},
                "adapter_native": {"correct": i < 8, "model_calls": 2, "elapsed_s": 1},
                "adapter_serializer": {"correct": i < 10, "model_calls": 3, "elapsed_s": 1.5},
            }
        })
    return {
        "model_digest": BASE_DIGEST,
        "adapter_sha256": sha256(b"adapter").hexdigest(),
        "source_manifest_sha256": sha256(b"manifest").hexdigest(),
        "tasks": tasks,
    }

def test_counts_and_interaction_are_not_promotion():
    report = audit(cohort())
    assert report["status"] == "DESCRIPTIVE_AUDIT_ONLY"
    assert report["paired"]["base_native -> adapter_native"]["rescues"] == 4
    assert report["paired"]["base_native -> adapter_native"]["regressions"] == 0
    assert report["interaction_net_cases"] == 0
    assert report["promotion_allowed"] is False
    assert report["aa_variance_checked"] is False

def test_blocks_duplicate_tasks():
    payload = cohort()
    payload["tasks"][1]["task_sha256"] = payload["tasks"][0]["task_sha256"]
    assert audit(payload)["status"] == "BLOCKED"

def test_blocks_one_family_and_partial_arms():
    payload = cohort()
    for row in payload["tasks"]:
        row["family"] = "only_source"
    assert audit(payload)["status"] == "BLOCKED"
    payload = cohort()
    del payload["tasks"][0]["arms"]["adapter_serializer"]
    assert audit(payload)["status"] == "BLOCKED"

def test_exact_p_for_six_zero():
    assert _mcnemar(6, 0) == 0.03125
