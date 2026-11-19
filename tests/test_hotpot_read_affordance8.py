from pathlib import Path

from scripts.run_hotpot_read_affordance8 import AffordanceSearch, LocalSearch, GATE


def test_affordance_adds_only_read_path_from_existing_original_hits(tmp_path):
    idx={"London":"pages/123456789abc.txt","Madrid":"pages/0123456789ab.txt"}
    files={"pages/123456789abc.txt":"London is a city",
           "pages/0123456789ab.txt":"Madrid is a city"}
    original=LocalSearch(idx,files).run({"query":"London"},tmp_path)
    treatment=AffordanceSearch(idx,files).run({"query":"London"},tmp_path)
    assert original.ok and treatment.ok
    assert treatment.content.startswith(original.content+"\n")
    assert 'read({"source": "pages/123456789abc.txt"})' in treatment.content
    assert 'pages/0123456789ab.txt' not in treatment.content
    assert "London is a city" in original.content


def test_no_found_hits_does_not_fabricate_source(tmp_path):
    idx={"London":"pages/123456789abc.txt"}
    files={"pages/123456789abc.txt":"some data"}
    original=LocalSearch(idx,files).run({"query":"zzz_not_here"},tmp_path)
    treatment=AffordanceSearch(idx,files).run({"query":"zzz_not_here"},tmp_path)
    assert treatment.content==original.content
    assert "read(" not in treatment.content


def test_effect_gate_requires_accuracy_and_actual_read_activation():
    assert GATE["min_net_normalized_correct"]>=2
    assert GATE["min_net_successful_read_tasks"]>=3
    assert GATE["max_normalized_regressions"]<=1
