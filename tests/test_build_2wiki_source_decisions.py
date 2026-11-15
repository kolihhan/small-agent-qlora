from hashlib import sha256

from scripts.build_2wiki_source_decisions import source_candidate, audit


def fixture():
    titles=["Film One","Person Two","Person Three","Person Four","Unused Five"]
    content=[]
    for i,title in enumerate(titles):
        next_name=titles[i+1] if i<3 else None
        head=("An original encyclopedia article. " + "details " * 15)
        body=(head + (f" It links to {next_name}." if next_name else
                     " Verified answer is Hannah Final."))
        content.append([body])
    return {
        "id":"source-test-1",
        "question":"Who was the final relative of Film One?",
        "answer":"Hannah Final",
        "type":"compositional",
        "context":{"title":titles,"sentences":content},
        "supporting_facts":{"title":titles[:4],"sent_id":[0,0,0,0]},
    }


def test_real_search_and_real_read_observations_create_nonoracle_eight_steps(tmp_path):
    candidate, reason=source_candidate(fixture(),tmp_path)
    assert reason=="candidate",reason
    assert candidate is not None
    assert len(candidate["source_support_titles"])==4
    assert candidate["real_tool_calls"]==8
    assert len(candidate["messages"])==19
    assert candidate["messages"][-1]=={"role":"assistant","content":"Hannah Final"}
    assistant=[m for m in candidate["messages"] if m["role"]=="assistant"]
    assert [m["tool_calls"][0]["function"]["name"] for m in assistant[:-1]] == (
        ["search","read"]*4
    )
    assert all(x["source_visible_before_search"] for x in candidate["transition_checks"])
    assert all(x["new_support_span_delivered"] for x in candidate["transition_checks"])
    assert not candidate["verified"] and not candidate["training_ready"]
    assert candidate["candidate_only"] and not candidate["native_agent_success_verified"]


def test_reference_answer_in_question_never_produces_teacher(tmp_path):
    row=fixture()
    row["question"]+=" Hannah Final"
    assert source_candidate(row,tmp_path)[1]=="answer_in_question_or_invalid"


def test_no_gold_target_forced_when_title_not_visible(tmp_path):
    row=fixture()
    row["context"]["sentences"][0]=["A long unrelated article with no reference to another page. "+"padding "*45]
    candidate,reason=source_candidate(row,tmp_path)
    assert candidate is None
    assert reason=="incomplete_or_nonminimal_evidence_path"


def test_streaming_audit_exports_only_candidate_never_verified(tmp_path):
    # Small unit audit verifies export structure even if pre-screen finds no
    # eligible original long paths. Production source is original 2Wiki TRAIN.
    path=tmp_path/"train-candidates.jsonl"
    result=audit([fixture()],path)
    assert result["original_first_n"]==1
    assert result["model_training_steps"]==0
    assert result["training_ready"] is False
    assert result["exported_candidate_only"] is True
    assert result["output_sha256"]==sha256(path.read_bytes()).hexdigest()
