from pathlib import Path

from scripts.eval_2wiki_native_agent import LocalSearch, freeze, page_path


def case(i):
    titles = [f"Alpha{i}", f"Beta{i}", f"Gamma{i}", f"Delta{i}", f"Other{i}", f"Unused{i}"]
    sentences = [
        [f"{title} has a published entity relation to Target{i}."]
        for title in titles
    ]
    return {
        "id": f"val-{i}", "question": f"What is Target{i} associated with?",
        "answer": f"Target{i}",
        "context": {"title": titles, "sentences": sentences},
        "supporting_facts": {"title": titles[:4], "sent_id": [0, 0, 0, 0]},
    }


def test_conditional_validation_freeze_seals_labels_in_record_not_model_prompt():
    sample = [case(i) for i in range(8)]
    frozen = freeze(sample)
    assert len(frozen["cases"]) == 8
    assert frozen["split"] == "validation"
    assert frozen["source_revision"]
    assert all(len(x["support"]) == 4 and len(x["files"]) == 6 for x in frozen["cases"])


def test_local_search_finds_real_source_file_and_preserves_distractors():
    titles = {"Alpha River": page_path("Alpha River"), "Other Hill": page_path("Other Hill")}
    files = {titles["Alpha River"]: "Alpha is an original article",
             titles["Other Hill"]: "Different original article"}
    tool = LocalSearch(titles, files)
    response = tool.run({"query": "Alpha River"}, Path("/tmp"))
    assert response.ok and titles["Alpha River"] in response.content
    assert titles["Other Hill"] not in response.content
    assert tool.validate_arguments({"query": "Alpha River"}) is None


def test_search_never_reveals_hidden_reference():
    titles = {"Alpha River": page_path("Alpha River")}
    files = {titles["Alpha River"]: "Alpha source content without the answer"}
    result = LocalSearch(titles, files).run({"query": "secret-answer"}, Path("/tmp"))
    assert "No matching" in result.content
