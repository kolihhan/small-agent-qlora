import pytest
import scripts.freeze_nextsearch_balanced_4k as selector
from test_build_nextsearch_causal_windows import parent_task
from scripts.build_nextsearch_causal_windows import state_windows


class NativeLikeTokenizer:
    def apply_chat_template(
        self,messages,tools,tokenize,add_generation_prompt,enable_thinking
    ):
        rendered="".join(
            f"<{m['role']}>:"+str(m.get("content") or "")+
            " "+str(m.get("tool_calls") or "")+"\n"
            for m in messages
        )
        if add_generation_prompt:
            rendered+="<assistant>:"
        return rendered
    def __call__(self,text,add_special_tokens=False):
        return {"input_ids":list(range(len(text.split())))}


def test_balanced_frozen_source_states_preserve_genuine_last_action(monkeypatch):
    monkeypatch.setattr(selector,"PER_STATE",1)
    rows,_=state_windows(parent_task())
    selected,report=selector.freeze(rows,NativeLikeTokenizer())
    assert len(selected)==2
    assert report["selected_actual_source_next_actions"]==2
    assert report["paired_source_parent_tasks"]==1
    assert report["all_retain_full_prompt"]
    assert report["states"]=={
        "evidence_missing_continue":1,
        "evidence_present_finalize":1
    }
    assert report["example_action_types"]=={"tool_call":1,"final":1}
    assert selected[0]["source_parent_sha256"]==selected[1]["source_parent_sha256"]


def test_rejects_state_assigned_after_evidence_is_visible(monkeypatch):
    monkeypatch.setattr(selector,"PER_STATE",1)
    rows,_=state_windows(parent_task())
    rows[0]["source_answer_visible_in_window"]=True
    with pytest.raises(ValueError,match="label contradicts"):
        selector.freeze(rows,NativeLikeTokenizer())


def test_does_not_silently_train_one_state_only(monkeypatch):
    monkeypatch.setattr(selector,"PER_STATE",1)
    rows,_=state_windows(parent_task())
    with pytest.raises(ValueError,match="not enough true original"):
        selector.freeze(rows[:1],NativeLikeTokenizer())


def test_rejects_nontraining_source_even_when_answer_correct(monkeypatch):
    monkeypatch.setattr(selector,"PER_STATE",1)
    rows,_=state_windows(parent_task())
    rows[0]["source_split"]="validation"
    with pytest.raises(ValueError,match="invalid source split"):
        selector.freeze(rows,NativeLikeTokenizer())
