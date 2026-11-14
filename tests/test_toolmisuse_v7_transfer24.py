import pytest

from scripts.toolmisuse_v7_transfer24 import DOMAINS, FAULTS, N_SHARDS, aggregate


def shard(arm, index):
    records = []
    # Each shard receives the same 6 pair positions under both arms.
    pairs = [(d, f) for d in DOMAINS for f in FAULTS]
    for k in range(index, len(pairs), N_SHARDS):
        d, fault = pairs[k]
        records.append({
            "id_hash": f"id-{k}", "domain": d, "fault": fault,
            "success": (k % 2 == 0 if arm == "base" else k % 3 == 0),
            "recovery_success": k % 6 == 0,
            "fault_seen": fault != "clean", "invalid_calls": 0,
            "tool_calls": 2, "budget_exceeded": False,
            "policy_violations": 0, "error": None,
        })
    return {
        "shard": index, "arm": arm, "source_sha": "dataset-sha",
        "selection_sha": "frozen24", "records": records,
    }


def test_transfer_aggregate_is_paired_and_reports_gains():
    parts = [shard(arm, i) for arm in ("base", "v7") for i in range(N_SHARDS)]
    r = aggregate(parts)
    assert r["n"] == 24
    assert r["arms"]["base"]["correct"] == 12
    assert r["arms"]["v7"]["correct"] == 8
    assert r["rescues"] > 0 and r["regressions"] > 0
    assert len(r["arms"]["base"]["by_domain"]) == 4
    assert len(r["arms"]["base"]["by_fault"]) == 6


def test_transfer_fails_closed_on_incomplete_or_changed_tasks():
    parts = [shard(arm, i) for arm in ("base", "v7") for i in range(N_SHARDS)]
    with pytest.raises(ValueError, match="paired"):
        aggregate(parts[:-1])
    parts[-1]["records"][0]["id_hash"] = "different"
    with pytest.raises(ValueError, match="paired task"):
        aggregate(parts)
