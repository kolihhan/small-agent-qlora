import json
from pathlib import Path

from gaia_small_agent.tools import default_tools
from gaia_small_agent.tools.workspace_find import WorkspaceFindTool
from scripts.eval_workspace_find_dev24 import N, select_rows, workspace_task


def test_opt_in_only_and_real_multihop_locator(tmp_path):
    names=[t.name for t in default_tools()]
    assert names==["search","read","inspect","python"]
    enabled=default_tools(workspace_find_enabled=True)
    assert [t.name for t in enabled]==names+["workspace_find"]
    root=tmp_path/"passages"
    root.mkdir()
    (root/"index.txt").write_text("p1 Alice biography\np2 town history")
    (root/"p1.txt").write_text("Alice moved to Princeton from Oxford in 1900.")
    (root/"p2.txt").write_text("Princeton belongs to New Jersey.")
    tool=WorkspaceFindTool()
    a=tool.run({"query":"Princeton"},tmp_path)
    assert a.ok and "passages/p1.txt" in a.content
    assert "passages/p2.txt" in a.content
    assert a.metadata["search_scope"]=="workspace_only"


def test_workspace_find_no_absolute_paths_symlink_or_network(tmp_path):
    outside=tmp_path.parent/"not_in_workspace.txt"
    outside.write_text("UNIQUE-OUTSIDE-MARKER")
    (tmp_path/"safe.txt").write_text("Princeton within workspace")
    (tmp_path/"outside.txt").symlink_to(outside)
    result=WorkspaceFindTool().run({"query":"UNIQUE-OUTSIDE-MARKER"},tmp_path)
    assert result.ok and "outside.txt" not in result.content and result.metadata["hits"]==0
    assert WorkspaceFindTool().run({"query":"P"},tmp_path).error_code=="BAD_ARGUMENTS"


def test_workspace_find_bounded_and_zero_matches(tmp_path):
    for idx in range(20):
        (tmp_path/f"x{idx}.txt").write_text("Name: New York, New York\n"*20)
    r=WorkspaceFindTool().run({"query":"New York","max_hits":3},tmp_path)
    assert r.ok and r.metadata["hits"]==3
    assert len(r.content.splitlines())==3
    assert WorkspaceFindTool().run({"query":"never-mentioned"},tmp_path).metadata["hits"]==0


def test_frozen_dev_selection_deterministic_without_gold_tuning():
    rows=[]
    for hop in (3,4):
        for k in range(20):
            idx=f"{hop}hop000__test{k}"
            rows.append({
                "id":idx, "answerable":True,
                "question":f"Question {idx}?", "answer":"answer",
                "question_decomposition":[{"id":f"sub{n}"} for n in range(hop)],
                "paragraphs":[{"idx":i,"title":f"Topic{i}",
                  "paragraph_text":"Wikipedia article passage about one named subject."} for i in range(20)],
            })
    a=select_rows(rows)
    b=select_rows(reversed(rows))
    assert len(a)==N==24
    assert [r["id"] for r in a]==[r["id"] for r in b]
    assert [r["id"][0] for r in a].count("3")==12
    assert [r["id"][0] for r in a].count("4")==12
    assert len(workspace_task(a[0])["files"])==21
