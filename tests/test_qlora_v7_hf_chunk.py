"""Source-disjoint, non-replayed 22-chunk v7 free ZeroGPU training contract."""
from collections import Counter
import pytest
from scripts.qlora_v7_hf_chunk import build_free_gpu_chunk


def make_rows():
    rows=[]
    for i in range(28):
        rows.append({"task_id":f"h-{i}", "source":"hotpotqa-distractor-train",
                     "training_action_indices":[2,4,6,8]})
    for i in range(16):
        rows.append({"task_id":f"w-{i}", "source":"2wikimultihopqa-train",
                     "training_action_indices":[2,4,6,8]})
    caps=("no_tool_stop","tool_selection","argument_grounding",
          "multi_step","evidence_to_final","failure_recovery",
          "strategy_switch","duplicate_avoidance")
    for j,cap in enumerate(caps):
        for i in range(21 if j>=6 else 22):
            rows.append({"task_id":f"s-{cap}-{i}", "source":"synthetic-small-agent-policy-v3/train",
                         "capability":cap, "training_action_indices":[2,4]})
    return rows


def test_22_chunks_cover_each_parent_once_no_gaia_replay():
    rows=make_rows()
    chunks=[build_free_gpu_chunk(rows, i) for i in range(22)]
    parents=[r["task_id"] for c in chunks for r in c]
    assert len(parents)==len(set(parents))==218
    assert set(parents)=={r["task_id"] for r in rows}
    assert sum(sum(len(r["training_action_indices"]) for r in c) for c in chunks)==524
    assert all(8<=len(c)<=10 for c in chunks)
    assert any(r["source"].startswith("2wikimultihopqa") for r in chunks[0])
    assert any(r["source"].startswith("hotpotqa") for r in chunks[0])
    assert all("no_tool_stop" in {r.get("capability") for r in c} for c in chunks[:21])


def test_chunk_selection_deterministic_order_invariant():
    a=make_rows()
    b=list(reversed(a))
    for i in (0,1,7,16,21):
        assert build_free_gpu_chunk(a,i)==build_free_gpu_chunk(b,i)


def test_refuse_invalid_chunk_or_unexpected_parent_sources():
    r=make_rows()
    with pytest.raises(ValueError,match="batch_index"):
        build_free_gpu_chunk(r,22)
    with pytest.raises(ValueError,match="source"):
        build_free_gpu_chunk([x for x in r if x["source"]!="2wikimultihopqa-train"],0)
    dup=r+[dict(r[0])]
    with pytest.raises(ValueError,match="duplicate"):
        build_free_gpu_chunk(dup,0)
