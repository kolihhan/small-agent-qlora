"""Generate *candidate* evidence-conditioned Qwen tool trajectories from real 2Wiki TRAIN.

The navigation selector is deliberately independent of publisher gold facts:
each next title must literally occur in the question or already delivered
tool text. Gold supporting-fact annotations are applied POST-HOC to reject
unhelpful or unsupported navigation. No fabricated search/read output.

These are untrained scripted navigation candidates, not an expert policy.
A later native Qwen OOD test is required BEFORE using them for QLoRA SFT.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from itertools import islice
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from gaia_small_agent.agent.loop import MAX_TOOL_OBSERVATION_CHARS, SYSTEM_PROMPT
from gaia_small_agent.tools import ReadTool
from gaia_small_agent.training.trajectories import _materialize_files
from scripts.audit_2wiki_search_read_paths import context_support, replay, visible
from scripts.eval_2wiki_native_agent import LocalSearch, page_path

REPO = "framolfese/2WikiMultihopQA"
REV = "fe713bfbd1afbca1a65246741a75890405d56a3a"
PROTOCOL = "2wiki-actual-tool-evidence-decision-candidates-v1"
CAP = 5000


def rank(title: str) -> str:
    return sha256(title.encode("utf-8")).hexdigest()


def tool_turn(name: str, arguments: dict, content: str, count: int) -> list[dict]:
    callid = f"call_{count}"
    return [
        {
            "role": "assistant", "content": "",
            "tool_calls": [{
                "id": callid, "type": "function",
                "function": {"name": name, "arguments": arguments}
            }],
        },
        {"role": "tool", "tool_call_id": callid, "name": name, "content": content}
    ]


def source_candidate(row: dict, root: Path) -> tuple[dict | None, str]:
    pages, support = context_support(row)
    if len(support) != 4 or len(pages) < 5:
        return None, "not_four_support"
    answer = str(row.get("answer") or "").strip()
    question = str(row["question"])
    if not 1 <= len(answer) <= 80 or answer.casefold() in question.casefold():
        return None, "answer_in_question_or_invalid"
    source_id = str(row.get("id") or row.get("_id") or "")
    if not source_id:
        return None, "missing_id"
    files = {page_path(title): content for title, content in pages.items()}
    _materialize_files({"files": files}, root)
    finder = LocalSearch({title: page_path(title) for title in pages}, files)
    reader = ReadTool()
    prompt = (
        "Answer the following question using the local article search and read tools. "
        "Search for article names and read returned paths before answering. "
        "A local source may be a distractor; decide whether more evidence is needed. "
        "Return only the answer, not reasoning.\n\nQuestion: " + question
    )
    messages = [{"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":prompt}]
    seen = prompt
    visited, observed_support, transitions = [], {}, []
    exposed_before_last = False
    tool_count = 0
    for _ in range(12):
        # The candidate policy sees ONLY question and delivered tool text.
        candidates = [title for title in pages
                      if title not in visited and visible(title, seen)]
        if not candidates:
            break
        title = sorted(candidates, key=rank)[0]
        if title not in support:
            return None, "distractor_selected"
        source = page_path(title)
        search = finder.run({"query": title}, root)
        if not search.ok:
            return None, "search_error"
        search_obs,_ = search.bounded_observation(MAX_TOOL_OBSERVATION_CHARS)
        if source not in search_obs:
            return None, "source_path_not_delivered"
        messages.extend(tool_turn("search", {"query":title}, search_obs,tool_count))
        tool_count += 1
        seen += "\n" + search_obs
        if any(answer.casefold() in seen.casefold()
               for _ in (0,)) and len(visited) < 3:
            exposed_before_last = True
        read = reader.run({"source":source}, root)
        if not read.ok:
            return None, "read_error"
        read_obs,_ = read.bounded_observation(MAX_TOOL_OBSERVATION_CHARS)
        before = seen
        spans = support[title]
        if not spans or not all(span.casefold() in read_obs.casefold() for span in spans):
            return None, "published_support_not_in_actual_bounded_read"
        if all(span.casefold() in before.casefold() for span in spans):
            return None, "read_did_not_add_support"
        messages.extend(tool_turn("read", {"source":source}, read_obs,tool_count))
        tool_count += 1
        seen += "\n" + read_obs
        visited.append(title)
        observed_support[title] = len(spans)
        transitions.append({
            "source":source, "title":title,
            "count_sources_before":len(visited)-1,
            "source_visible_before_search":visible(title,before),
            "search_path_observed":True,
            "new_support_span_delivered":True,
            "answer_observed_before_read":answer.casefold() in before.casefold(),
            "answer_observed_after_read":answer.casefold() in seen.casefold()
        })
        if len(visited) == len(support):
            break
    if len(visited) != 4 or set(visited) != set(support):
        return None, "incomplete_or_nonminimal_evidence_path"
    if len(messages) != 2 + len(visited) * 4:
        raise AssertionError("every source requires one real search and read")
    if exposed_before_last:
        return None, "answer_visible_too_early"
    if answer.casefold() not in seen.casefold():
        return None, "answer_not_supported_by_delivered_content"
    if not transitions[-1]["answer_observed_after_read"]:
        return None, "no_answer_after_last_read"
    # This is a PUBLISHER-LABELED stopping point, not a learned Qwen stop.
    # We do not label the final action unless last read adds answer evidence
    # and the answer was absent earlier.
    decisive = (
        transitions[-1]["answer_observed_after_read"] and
        not transitions[-1]["answer_observed_before_read"]
    )
    if not decisive:
        return None, "last_read_not_answer_decisive"
    messages.append({"role":"assistant","content":answer})
    if len([m for m in messages if m["role"]=="assistant" and m.get("tool_calls")]) != 8:
        raise AssertionError("search and read count drift")
    return {
        "task_id":source_id,
        "source":REPO,"source_revision":REV,"source_split":"train",
        "source_license":"apache-2.0",
        "task_type":str(row.get("type") or ""),
        "provenance":{
            "license":"Apache-2.0",
            "generator":REPO,
            "generator_version":REV,
            "oracle_type":"visible-original-source-navigation-plus-posthoc-publisher-support",
            "oracle_version":PROTOCOL
        },
        "source_support_titles":visited,
        "source_support_sha256":[sha256(pages[t].encode()).hexdigest() for t in visited],
        "question_sha256":sha256(question.encode()).hexdigest(),
        "tools":[finder.definition(),reader.definition()],
        "messages":messages,
        "training_action_indices":[
            i for i,m in enumerate(messages)
            if m["role"]=="assistant"
        ],
        "transition_checks":transitions,
        "real_tool_calls":tool_count,
        "source_complete":True,
        "answer_source_decisive":decisive,
        # Never claim the scripted selector is actual Qwen expert behavior.
        "native_agent_success_verified":False,
        "verified":False,
        "training_ready":False,
        "candidate_only":True,
    }, "candidate"


def audit(rows: list[dict], output: Path) -> dict:
    if not rows:
        raise ValueError("empty original source cohort")
    counts=Counter()
    task_ids=set()
    trained_titles=set()
    export=[]
    with TemporaryDirectory(prefix="2wiki-candidate") as tmp:
        root=Path(tmp)
        for row in rows:
            cid=str(row.get("id") or row.get("_id") or "")
            if not cid or cid in task_ids:
                raise ValueError("missing or duplicate TRAIN source ID")
            task_ids.add(cid)
            preview=replay(row)
            if not preview["long_candidate"]:
                counts["preflight_not_long_path"]+=1
                continue
            counts["source_path_candidates"]+=1
            taskroot=root/sha256(cid.encode()).hexdigest()[:14]
            try:
                candidate,reason=source_candidate(row,taskroot)
            except (ValueError,OSError) as e:
                candidate,reason=None,"invalid_source_or_tool_bytes"
            counts[reason]+=1
            if candidate:
                # Keep train parents source-title disjoint so a single popular
                # Wikipedia biography cannot dominate learned search policy.
                if trained_titles & set(candidate["source_support_titles"]):
                    counts["overlapping_support_source"]+=1
                    continue
                trained_titles.update(candidate["source_support_titles"])
                export.append(candidate)
    with output.open("x",encoding="utf-8") as f:
        for row in export:
            f.write(json.dumps(row,ensure_ascii=False)+"\n")
    return {
        "protocol":PROTOCOL,"original_source":REPO,
        "revision":REV,"split":"train",
        "original_first_n":len(rows),"gold_action_selection":False,
        "local_search_tool_includes_original_distractors":True,
        "bounded_observation_chars":MAX_TOOL_OBSERVATION_CHARS,
        "source_visible_only_navigation_candidates":counts["source_path_candidates"],
        "fully_supported_8tool_plus_1final_candidates":len(export),
        "total_nonredundant_actual_tool_calls":sum(r["real_tool_calls"] for r in export),
        "output_jsonl":str(output),"output_sha256":sha256(output.read_bytes()).hexdigest(),
        "task_type_counts":dict(Counter(r["task_type"] for r in export)),
        "rejection_counts":dict(counts),
        "model_training_steps":0,
        "model_agent_steps":0,
        "exported_candidate_only":True,
        "training_ready":False,
        "limitations":"Local source search is scripted, not expert Qwen; posterior support labels filter trajectories; final answer relies on source QA reference; source may overlap pretraining. Must pass native policy and held-out task checks before training."
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--limit",type=int,default=CAP)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or a.report.exists() or not 1<=a.limit<=CAP:
        raise ValueError("refuse overwrite/invalid source limit")
    from datasets import load_dataset
    rows=list(islice(load_dataset(REPO,split="train",revision=REV,streaming=True),a.limit))
    if len(rows)!=a.limit:
        raise ValueError("original dataset truncated")
    result=audit(rows,a.output)
    print("2WIKI_SOURCE_GROUNDED_TRAIN_CANDIDATES",json.dumps(result,sort_keys=True),flush=True)
    a.report.write_text(json.dumps(result,indent=2)+"\n")


if __name__ == "__main__":
    main()
