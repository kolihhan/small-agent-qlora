from scripts.audit_v8_final_answer_priority import summarize
import pytest


def fixture():
    return {
        "n":100, "run_id":44980055775,"base_correct":20,"v8_correct":21,
        "changed_15_cases":[
            {"task_sha256_16":str(i),"outcome":"rescue" if i<8 else "regression",
             "base":{"correct":i>=8,"stop_reason":"invalid_final" if i<4 else "final"},
             "v8":{"correct":i<8,"stop_reason":"invalid_final" if i==8 else "final",
                   "failure_label":"duplicate_action_present" if i==8 else "incorrect_after_tools" if i>=8 else "correct"}}
            for i in range(15)
        ],
    }


def test_v8_regressions_are_mostly_normal_final_wrong_answers():
    r=summarize(fixture())
    assert r["v8_regressions_normal_final_but_wrong"]==6
    assert r["v8_regressions_eligible_for_final_text_retry"]==1
    assert r["rescue_source_stop_types"]["base_stop_invalid_final"]==4
    assert not r["no_new_score"] is False
    assert r["gold_or_answers_accessed"] is False


def test_audit_fails_on_incomplete_arms():
    bad=fixture()
    bad["changed_15_cases"].pop()
    with pytest.raises(ValueError,match="Discordant"):
        summarize(bad)
