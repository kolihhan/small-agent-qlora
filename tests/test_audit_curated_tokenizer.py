"""Offline-only, fake tokenizer tests for real QLoRA native prompt budgets."""
import pytest

from scripts.audit_curated_tokenizer import analyze_examples


class StubTokenizer:
    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids":list(text)}


def row(task,action,prompt_length,completion_length):
    return {"prompt":"p"*prompt_length,"completion":"c"*completion_length,
            "metadata":{"task_id":task,"action_type":action}}


def test_real_context_budget_is_measurement_not_silent_truncation():
    samples=[
        row("hot","tool_call",700,200),
        row("hot","final",1000,80),
        row("wiki","tool_call",1500,350),
    ]
    r=analyze_examples(samples,StubTokenizer(),
                       {"hot":"hotpotqa-distractor-train","wiki":"2wikimultihopqa-train"})
    assert r["native_train_actions"]==3 and r["unique_tasks"]==2
    assert r["fits_t4_1024"]==1
    assert r["fits_2048"]==3
    assert r["max_total_tokens"]==1850
    assert r["source_token_fit"]["hotpotqa-distractor-train"]["actions"]==2
    assert r["action_type_counts"]=={"final":1,"tool_call":2}
    assert r["fully_supported_tasks_at_1024"]==0
    assert r["fully_supported_tasks_at_2048"]==1
    assert r["tasks_with_final_action"]==1
    assert r["final_actions_fit_1024"]==0
    assert r["final_actions_fit_2048"]==1
    assert r["action_type_token_fit"]["final"]["fits_2048"]==1
    assert r["untruncated_training_ran"] is False
    assert r["gaia_evaluated"] is False


def test_action_not_attributed_to_known_source_fails_closed():
    with pytest.raises(ValueError,match="unattributed"):
        analyze_examples([row("unknown","tool_call",30,5)],StubTokenizer(),
                         {"known":"hotpotqa-distractor-train"})


def test_zero_completion_and_empty_source_are_not_accepted():
    with pytest.raises(ValueError,match="No rendered"):
        analyze_examples([],StubTokenizer(),{})
    with pytest.raises(ValueError,match="empty completion"):
        analyze_examples([row("known","final",20,0)],StubTokenizer(),
                         {"known":"2wikimultihopqa-train"})
