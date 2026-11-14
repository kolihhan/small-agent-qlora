import json

import pytest

from gaia_small_agent.training.oracle_policy_v3 import generate_oracle_policy_v3_dataset
from scripts.gaia_format_reminder_ablation import FormatModel, aggregate, select_cases


def test_format_probe_is_deterministic_and_excludes_previously_inspected_three(tmp_path):
    source = generate_oracle_policy_v3_dataset(tmp_path)["test"]
    cohort = select_cases(source)
    source_rows = [json.loads(line) for line in source.read_text().splitlines()]
    assert len(cohort) == 24
    for cap in ("failure_recovery", "strategy_switch"):
        seen_first_three = {r["task_id"] for r in source_rows if r["capability"] == cap} 
        first = [r["task_id"] for r in source_rows if r["capability"] == cap][:3]
        assert not set(first).intersection(r["task_id"] for r in cohort)
    assert [r["task_id"] for r in cohort] == [r["task_id"] for r in select_cases(source)]


def test_format_model_does_not_mutate_original_transcript():
    base = FormatModel(reminder=False)
    fmt = FormatModel(reminder=True)
    messages = [{"role": "system", "content": "base"}, {"role": "user", "content": "q"}]
    assert base._prepare(messages) == messages
    adjusted = fmt._prepare(messages)
    assert len(adjusted) == len(messages)
    assert adjusted[1] == messages[1]
    assert adjusted[0]["content"].startswith("base")
    assert "output format exactly" in adjusted[0]["content"]
    assert messages[0]["content"] == "base"


def test_format_aggregate_rejects_missing_shard():
    with pytest.raises(ValueError, match="missing/duplicate"):
        aggregate([])
