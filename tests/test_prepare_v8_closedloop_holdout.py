from hashlib import sha256
import json

import pytest
from scripts.prepare_v8_closedloop_holdout import _full_files, _observed_reads, recover


def _fixture(i):
    ident = f"4hop3__fixture_{i}"
    src = {"id": ident, "question": f"Where is the location {i}?", "answer": f"Answer-{i}",
           "paragraphs": [{"idx": j, "title": f"Topic {j}",
                           "paragraph_text": f"Topic {j} evidence for fixture {i} " + ("a" * 25)}
                          for j in range(20)]}
    files = _full_files(src)
    messages = [{"role": "system", "content": "system"},
                {"role": "user", "content": "Question: " + src["question"]}]
    for j, name in enumerate(("passages/index.txt", "passages/p1.txt", "passages/p2.txt", "passages/p3.txt")):
        call_id = f"r{j}"
        messages.append({"role": "assistant", "content": "", "tool_calls": [{
            "id": call_id, "function": {"name": "read", "arguments": {"source": name}}}]})
        body = files[name]
        messages.append({"role": "tool", "tool_call_id": call_id, "name": "read",
                         "content": f'ToolMeta: {json.dumps({"offset_chars":0, "returned_chars":len(body)})}\n{body}'})
    messages.append({"role": "assistant", "content": src["answer"]})
    hold = {"task_id": "musique-train-" + ident, "source": "musique-train", "verified": True,
            "messages": messages, "expected_answer": src["answer"], "hop_count": 4,
            "source_id_sha256": sha256(ident.encode()).hexdigest()}
    return src, hold


def test_full_workspaces_source_checked_without_teacher_forcing():
    pairs = [_fixture(i) for i in range(24)]
    holdout = [x for _, x in pairs] + [
        {"task_id": f"other-{i}", "source": "hotpotqa-distractor-train"} for i in range(11)]
    train = [{"task_id": f"train-{i}"} for i in range(304)]
    tasks, audit = recover([x for x, _ in pairs], holdout, train)
    assert len(tasks) == 24 and audit["train_holdout_task_overlap"] == 0
    assert all(len(x["eval_task"]["files"]) == 21 for x in tasks)
    assert all("messages" not in x["eval_task"] for x in tasks)
    assert audit["tool_calls_teacher_forced"] is False


def test_physical_source_mismatch_is_not_silently_allowed():
    pairs = [_fixture(i) for i in range(24)]
    holdout = [x for _, x in pairs] + [
        {"task_id": f"other-{i}", "source": "hotpotqa-distractor-train"} for i in range(11)]
    train = [{"task_id": f"train-{i}"} for i in range(304)]
    holdout[0]["messages"][3]["content"] = holdout[0]["messages"][3]["content"].replace("Topic", "WRONG")
    with pytest.raises(ValueError, match="observation mismatch"):
        recover([x for x, _ in pairs], holdout, train)


def test_file_traversal_and_partial_source_rejected():
    src, row = _fixture(2)
    row["messages"][2]["tool_calls"][0]["function"]["arguments"]["source"] = "../secret"
    with pytest.raises(ValueError, match="unsafe target"):
        _observed_reads(row)
    with pytest.raises(ValueError, match="20 original source"):
        _full_files({"paragraphs": src["paragraphs"][:-1]})
