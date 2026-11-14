import pytest

from scripts.finqa_numeric_v7_transfer20 import number, score_numeric, aggregate


def test_numeric_tolerances_and_percent_scaling():
    assert number("1,234.50%") == (1234.5, True)
    assert score_numeric("0.935", "93.5%")
    assert score_numeric("93.5%", "93.5%")
    assert score_numeric("127.40", "127.4")
    assert not score_numeric("93.5", "123.5")
    assert not score_numeric("The answer is 0.935", "93.5%")


def fake(arm, i):
    return {
        "shard": i, "arm": arm, "source_sha": "dev-pinned",
        "cohort_hash": "cohort-v1",
        "rows": [{
            "id_hash": f"id-{j}", "correct": (j % 2 == 0 if arm == "base" else j % 3 == 0),
            "completed": True, "stop": "final", "tool_calls": 3,
            "tool_errors": 0, "duplicates": 0, "backend_error": None,
        } for j in range(i, 20, 5)],
    }


def test_aggregate_is_paired_and_complete():
    p = [fake(arm, i) for arm in ("base", "v7") for i in range(5)]
    result = aggregate(p)
    assert result["n"] == 20
    assert result["arms"]["base"]["correct"] == 10
    assert result["arms"]["v7"]["correct"] == 7
    assert result["rescues"] > 0 and result["regressions"] > 0


def test_aggregate_fails_closed_on_partial_and_mismatch():
    p = [fake(arm, i) for arm in ("base", "v7") for i in range(5)]
    with pytest.raises(ValueError, match="five shards"):
        aggregate(p[:-1])
    p[-1]["rows"][0]["id_hash"] = "wrong"
    with pytest.raises(ValueError, match="matched task identity"):
        aggregate(p)
