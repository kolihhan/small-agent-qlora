"""Source-only test: fixed 19 by 3 scenarios; no DEV data/model needed."""
from hashlib import sha256
from types import SimpleNamespace
import pytest

from scripts import eval_appworld_qwen9b_dev57 as p


def _tasks():
    return [f"scene{num:02d}_{episode}" for num in range(19) for episode in (1,2,3)]


def test_exactly_19_scene_shards_no_task_duplication():
    ids=_tasks()
    groups=p.arrange_scenarios(ids)
    assert len(groups)==19
    assert all(len(x)==3 for x in groups)
    assert sorted(x for group in groups for x in group)==sorted(ids)
    assert groups==p.arrange_scenarios(list(reversed(ids)))
    assert len({t.split("_",1)[0] for group in groups for t in group})==19


def test_missing_or_mixed_scene_shards_fail_closed():
    ids=_tasks()
    with pytest.raises(ValueError):
        p.arrange_scenarios(ids[:56])
    with pytest.raises(ValueError):
        p.arrange_scenarios(ids[:-1]+[ids[0]])
    with pytest.raises(ValueError):
        p.arrange_scenarios([f"x{i}_1" for i in range(57)])


def test_original_model_source_and_tool_policy_fixed():
    assert p.SHARDS==19 and p.MAX_STEPS==20 and p.MAX_API_CALLS==48
    assert p.STUDENT_POLICY.endswith("-v1")
    assert p.MODEL_GGUF_SHA.startswith("8c4cc24f")
    assert p.PROTOCOL.startswith("SMALL-AGENT-APPWORLD-DEV57")


def test_never_publish_partial_dev57_as_a_complete_score():
    with pytest.raises(ValueError):
        p.combine([],{},[])
    assert p.N==57
