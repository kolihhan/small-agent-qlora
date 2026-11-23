"""Synthetic MuSiQue source replay tests. No benchmark question/answers used."""
from tempfile import TemporaryDirectory
from pathlib import Path

from gaia_small_agent.agent.loop import MAX_TOOL_OBSERVATION_CHARS
from gaia_small_agent.tools import default_tools
from gaia_small_agent.training.trajectories import _materialize_files
from scripts.prepare_v9_real_source_recovery import augment, _files_from_source


def _case(with_link=True):
    files={
        "passages/index.txt":"passages/p5.txt | Example Topic\npassages/p9.txt | Distractor Topic",
        "passages/p5.txt":"Example Topic: the fictional answer is blue.",
        "passages/p9.txt":"Distractor Topic: irrelevant information."
    }
    reader=next(t for t in default_tools() if t.name=="read")
    with TemporaryDirectory() as tmp:
        root=Path(tmp)
        _materialize_files({"files":files},root)
        index,_=reader.run({"source":"passages/index.txt"},root).bounded_observation(MAX_TOOL_OBSERVATION_CHARS)
        passage,_=reader.run({"source":"passages/p5.txt"},root).bounded_observation(MAX_TOOL_OBSERVATION_CHARS)
    def call(source, ident):
        return {"role":"assistant","content":"","tool_calls":[
            {"id":ident,"type":"function","function":{"name":"read","arguments":{"source":source}}}]}
    messages=[
        {"role":"system","content":"Use tools"},
        {"role":"user","content":"Find Example Topic. Start with passages/index.txt."},
        call("passages/index.txt","a"),
        {"role":"tool","name":"read","tool_call_id":"a","content":index if with_link else "No listed passage title"},
        call("passages/p5.txt","b"),
        {"role":"tool","name":"read","tool_call_id":"b","content":passage},
        {"role":"assistant","content":"blue"},
    ]
    row={
        "source":"musique-train","task_id":"fictional-synthetic-test",
        "derived_from_annotation_not_observed_model":True,
        "annotation_supervision_profile":"visible_link_or_final_v1",
        "link_grounded_read_labels":1,
        "messages":messages,"training_action_indices":[2,4,6],
        "verification":{"zero_tool_errors":True},
        "metrics":{"tool_calls":2,"tool_errors":0}
    }
    return row,files


def test_verified_source_read_error_and_label_stability():
    row,files=_case()
    new=augment(row,files)
    assert new is not None
    assert new["training_action_indices"]==[2,6,8]
    assert new["messages"][5]["content"].startswith("ToolError[NOT_FOUND]: ")
    assert [new["messages"][i] for i in new["training_action_indices"]]==[
        row["messages"][i] for i in row["training_action_indices"]]
    assert new["verification"]["expected_tool_error_codes"]==["NOT_FOUND"]
    assert new["error_conditioned_augmentation"]["not_observed_agent_behavior"]
    assert new["metrics"]["tool_errors"]==1


def test_hidden_source_action_never_eligible():
    row,files=_case(with_link=False)
    assert augment(row,files) is None


def test_rejects_other_sources_and_wrong_original_response():
    row,files=_case()
    row["source"]="2wikimultihopqa-train"
    assert augment(row,files) is None
    row["source"]="musique-train"
    row["messages"][5]["content"]="fabricated response"
    try:
        augment(row,files)
    except ValueError as exc:
        assert "differs from replay" in str(exc)
    else:
        raise AssertionError("Expected fail-closed replay mismatch")


def test_pinned_raw_source_workspace_requires_20_distinct_parts():
    raw={"paragraphs":[{"idx":i,"title":f"Fictional {i}",
                        "paragraph_text":"Evidence from a fake passage."} for i in range(20)]}
    files=_files_from_source(raw)
    assert len(files)==21
    assert "passages/p0.txt" in files
    raw["paragraphs"][5]["idx"]=4
    try:
        _files_from_source(raw)
    except ValueError as exc:
        assert "Duplicate" in str(exc)
    else:
        raise AssertionError("Expected source collision rejection")
