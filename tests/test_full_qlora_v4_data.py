"""A real full-dataset QLoRA run must not be confused with 64-step v3E."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from scripts.prepare_full_qlora_v4 import prepare_full_clean_train


def test_full_training_removes_all_deliberately_bad_first_search_actions(tmp_path):
    from gaia_small_agent.training.oracle_policy_v3 import generate_oracle_policy_v3_dataset
    files = generate_oracle_policy_v3_dataset(tmp_path/"seed", seed="small-agent-policy-v3")
    source = [json.loads(line) for line in files["train"].read_text().splitlines()]
    cleaned, stats = prepare_full_clean_train(source)
    assert len(cleaned) == 1600
    assert len({r["task_id"] for r in cleaned}) == len(cleaned)
    assert stats["removed_intentionally_bad_actions"] == 400
    assert stats["total_trainable_actions"] == 3200
    counts = Counter(r["capability"] for r in cleaned)
    assert set(counts.values()) == {200}
    for row in cleaned:
        assert row["source"].endswith("/train")
        idx = row["training_action_indices"]
        assert idx
        assert all(row["messages"][i]["role"] == "assistant" for i in idx)
        assert len(set(idx)) == len(idx)
        if row["capability"] in {"failure_recovery", "strategy_switch"}:
            assert 2 not in idx
            assert 4 in idx
            assert 6 in idx
        else:
            assert 2 in idx or row["capability"] == "no_tool_stop"
        assert row["verification"]["completed"] is True


def test_full_training_rejects_eval_data_and_unverified_rows(tmp_path):
    from gaia_small_agent.training.oracle_policy_v3 import generate_oracle_policy_v3_dataset
    files = generate_oracle_policy_v3_dataset(tmp_path/"seed")
    row = json.loads(files["dev"].read_text().splitlines()[0])
    try:
        prepare_full_clean_train([row])
        assert False, "dev data must never enter SFT"
    except ValueError as e:
        assert "train split" in str(e)
    train = json.loads(files["train"].read_text().splitlines()[0])
    train["verified"] = False
    try:
        prepare_full_clean_train([train])
        assert False
    except ValueError as e:
        assert "verified" in str(e)
