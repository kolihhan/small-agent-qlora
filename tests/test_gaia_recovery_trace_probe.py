from gaia_small_agent.training.oracle_policy_v3 import generate_oracle_policy_v3_dataset
from scripts.gaia_recovery_trace_probe import select_probe_rows


def test_select_six_heldout_cases_without_changing_generator(tmp_path):
    source = generate_oracle_policy_v3_dataset(tmp_path)["test"]
    a = select_probe_rows(source)
    b = select_probe_rows(source)
    assert [r["task_id"] for r in a] == [r["task_id"] for r in b]
    assert [r["capability"] for r in a] == ["failure_recovery"] * 3 + ["strategy_switch"] * 3
    assert len({r["task_id"] for r in a}) == 6
