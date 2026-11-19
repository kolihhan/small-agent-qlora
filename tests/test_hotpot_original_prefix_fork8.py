from gaia_small_agent.agent.types import TraceEvent
from scripts.run_hotpot_original_prefix_fork8 import (
    PROTOCOL, OLD8_SHA, OLD16_SHA, N, check_blind,
    digest, first_read, freeze_new, normalize, read_paths,
)


def test_first_successful_actual_read_not_just_search():
    events = [
        TraceEvent("tool_call", 1, {"id":"s", "name":"search","arguments":{"query":"x"}}),
        TraceEvent("tool_result",1, {"id":"s","ok":True}),
        TraceEvent("tool_call",2,{"id":"r0","name":"read","arguments":{"source":"missing.txt"}}),
        TraceEvent("tool_result",2, {"id":"r0","ok":False}),
        TraceEvent("tool_call",3,{"id":"r1","name":"read","arguments":{"source":"pages/one.txt"}}),
        TraceEvent("tool_result",3, {"id":"r1","ok":True}),
    ]
    assert first_read(events) == 3
    assert read_paths(events,3) == ["pages/one.txt"]


def test_locked_answer_normalizer_does_not_invent_distractor():
    assert normalize("Discussion\nFinal Answer: **Sydney.**") == "Sydney"
    assert normalize("Answer: 183.") == "183"
    assert normalize("The evidence is insufficient") == "The evidence is insufficient"


def test_source_split_excludes_prior_24_and_labels():
    old8={"cases_sha256":OLD8_SHA,"n":8,
          "cases":[{"id":"prior8-"+str(i),"support":{"pages/old8": ["x"]}} for i in range(8)]}
    old16={"new_source_sha":OLD16_SHA,"n":16,
           "cases":[{"id":"prior16-"+str(i),"support":{"pages/old16": ["x"]}} for i in range(16)]}
    rows=[]
    for i in range(24):
        titles=[f"Article {i} {j}" for j in range(8)]
        rows.append({
            "id":"fresh-"+str(i), "type":"bridge" if i%2 else "comparison",
            "question":f"Who is the relevant person for event {i}?",
            "answer":f"Person {i}",
            "context":{"title":titles,"sentences":[[f"Article {j} information about event {i}." ] for j in range(8)]},
            "supporting_facts":{"title":titles[:2],"sent_id":[0,0]},
        })
    blind,ref=freeze_new(rows,old8,old16)
    assert blind["n"]==N and len(blind["cases"])==N
    assert len({x["id"] for x in blind["cases"]})==N
    assert all("answer" not in x and "support" not in x for x in blind["cases"])
    assert all("answer" in x and len(x["support_paths"])==2 for x in ref["references"])
    assert blind["cases_sha256"]==ref["cases_sha256"]
    check_blind(blind)
