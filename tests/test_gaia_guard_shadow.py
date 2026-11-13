from scripts.gaia_guard_shadow import shard_rows, aggregate


def test_shadow_frozen_25_divides_4_way():
    rows=[{"task_id":str(i)} for i in range(25)]
    assert [len(shard_rows(rows,index=i)) for i in range(4)]==[7,6,6,6]


def test_paired_shadow_metrics_and_no_content_exposed():
    shards=[]
    for shard in range(4):
        rows=[]
        for i in range(shard,25,4):
            for arm in ("baseline","guard"):
                rows.append({
                    "task_hash":f"hash-{i}","level":1,"arm":arm,
                    "correct": bool(i==0) if arm=="baseline" else bool(i in {0,1}),
                    "completed":True,"tool_calls":2,
                    "duplicate_calls_blocked":0,"tool_errors":0,"steps":3,
                })
        shards.append({"schema":"gaia-tool-guard-shadow-shard-v1","shard":shard,"rows":rows})
    agg=aggregate(shards)
    assert agg["task_count"]==25
    assert agg["baseline"]["correct"]==1
    assert agg["guard"]["correct"]==2
    assert agg["net_correct"]==1
    assert agg["new_training_ran"] is False
    assert all("Question" not in row and "Final answer" not in row for row in agg["rows"])
