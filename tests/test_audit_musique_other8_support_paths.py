from types import SimpleNamespace

import pytest
from scripts.audit_musique_other8_support_paths import supporting_path_stats


def _sample_source():
    return {"paragraphs":[{"idx":i,"is_supporting":i in (1,4,6)}
                          for i in range(20)]}


def test_annotated_paths_recall_is_posthoc_upper_bound_not_model_accuracy():
    source=_sample_source()
    obs={"read_paths":["passages/index.txt","passages/p1.txt","passages/p4.txt",
                       "passages/p4.txt","passages/p10.txt"],
         "correct":False,"completed":True}
    report=supporting_path_stats(source,obs)
    assert report["annotated_support_passages"]==[1,4,6]
    assert report["support_paths_visited"]==[1,4]
    assert report["missing_annotated_paths"]==[6]
    assert report["support_path_recall_upper_bound"]==0.6667
    assert report["distinct_paths_read"]==3
    assert report["all_support_paths_visited"] is False
    obs["read_paths"].append("passages/p6.txt")
    assert supporting_path_stats(source,obs)["all_support_paths_visited"] is True
    assert supporting_path_stats(source,obs)["correct"] is False


def test_invalid_annotation_or_nonsource_path_fail_closed():
    src=_sample_source()
    src["paragraphs"][3]["is_supporting"]="False"
    with pytest.raises(ValueError,match="support labels"):
        supporting_path_stats(src,{"read_paths":[]})
    src=_sample_source()
    with pytest.raises(ValueError,match="nonexistent paragraph"):
        supporting_path_stats(src,{"read_paths":["passages/p99.txt"]})
    with pytest.raises(ValueError,match="read-path log absent"):
        supporting_path_stats(src,{})
