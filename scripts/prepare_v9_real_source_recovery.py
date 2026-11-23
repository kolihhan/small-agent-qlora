"""Prepare v9 evidence-grounded *real-source* failure-conditioned QLoRA training.

Controlled change: preserve original 304 TRAIN parents and 798 selected
assistant targets. Inject a real ReadTool NOT_FOUND observation before ONE
originally source-visible, label-safe read action per eligible MuSiQue TRAIN
parent, using physical reconstruction from immutable MuSiQue source bytes.

Error is deliberately induced by a synthetic filename typo, NOT observed
autonomous agent behavior. No oracle answer, support flag or decomposition is
introduced into model input; the next-action target was already selected by
v8's visible-link gate. Training and GAIA evaluation are NOT performed here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from gaia_small_agent.agent.loop import MAX_TOOL_OBSERVATION_CHARS
from gaia_small_agent.tools import default_tools
from gaia_small_agent.training.trajectories import _materialize_files

TRAIN_SHA = "203a643b9ef809bad311a4adad0ef24caccd9505f3fb3f8981ab95da5d999ac7"
MUSIQUE_TRAIN_SHA = "83a75b1e11e4e9bb8f8308e72ac40ca617ae4431b3a0d955b61cab259248490a"
SEED = "v9-mu-verified-read-fail-recovery-20261123-frozen-1"
N_PARENTS, N_ACTIONS = 304, 798


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf8").splitlines() if x.strip()]


def _read_source(msg: dict) -> str | None:
    if msg.get("role") != "assistant":
        return None
    calls=msg.get("tool_calls") or []
    if len(calls)!=1:
        return None
    fn=calls[0].get("function") or {}
    if fn.get("name")!="read":
        return None
    args=fn.get("arguments") or {}
    source=args.get("source")
    return source if isinstance(source,str) and source.startswith("passages/") else None


def _files_from_source(source: dict) -> dict[str, str]:
    parts=source.get("paragraphs")
    if not isinstance(parts,list) or len(parts)!=20:
        raise ValueError("Original MuSiQue workspace missing 20 passages")
    files={}
    index=[]
    for p in sorted(parts,key=lambda p:p["idx"]):
        ix=p["idx"]
        if type(ix) is not int or not 0<=ix<100:
            raise ValueError("MuSiQue passage identity invalid")
        title=p.get("title","")
        body=p.get("paragraph_text")
        if not isinstance(title,str) or not isinstance(body,str) or not body:
            raise ValueError("MuSiQue passage content invalid")
        name=f"passages/p{ix}.txt"
        if name in files:
            raise ValueError("Duplicate passage identity")
        files[name]=body
        index.append(f"{name} | {title}")
    files["passages/index.txt"]="\n".join(index)
    if len(files)!=21:
        raise ValueError("Missing source passage identity")
    return files


def _visible_before(messages: list[dict], idx: int, path: str) -> bool:
    # Exact path must have already appeared in a user/tool message.
    seen="\n".join(str(m.get("content") or "") for m in messages[:idx]
                   if m.get("role") in ("user","tool"))
    return path in seen


def augment(row: dict, files: dict[str,str]) -> dict | None:
    if row.get("source")!="musique-train":
        return None
    if (row.get("derived_from_annotation_not_observed_model") is not True
        or row.get("annotation_supervision_profile")!="visible_link_or_final_v1"
        or int(row.get("link_grounded_read_labels") or 0)<1):
        return None
    messages=row.get("messages") or []
    selected=row.get("training_action_indices") or []
    if not isinstance(selected,list) or not selected:
        raise ValueError("Invalid source-annotated selected action indices")
    read=next(t for t in default_tools() if t.name=="read")
    for idx in selected:
        target=_read_source(messages[idx])
        if target is None or target=="passages/index.txt" or target not in files:
            continue
        if not _visible_before(messages,idx,target):
            continue
        if idx+1>=len(messages) or messages[idx+1].get("role")!="tool":
            continue
        suffix=sha256(f"{SEED}:{row['task_id']}:{idx}".encode()).hexdigest()[:12]
        bad=target+".missing-"+suffix
        if bad in files:
            raise ValueError("Injected missing path exists in authentic source")
        with TemporaryDirectory() as tmp:
            workspace=Path(tmp)
            _materialize_files({"files":files},workspace)
            failed=read.run({"source":bad},workspace)
            passed=read.run({"source":target},workspace)
            if failed.ok or failed.error_code!="NOT_FOUND" or not passed.ok:
                raise ValueError("Native ReadTool failed physical expected-result proof")
            obs,_=failed.bounded_observation(MAX_TOOL_OBSERVATION_CHARS)
            good,_=passed.bounded_observation(MAX_TOOL_OBSERVATION_CHARS)
            if good!=str(messages[idx+1].get("content") or ""):
                raise ValueError("Original true observation differs from replay: no safe intervention")
        if "ToolError[NOT_FOUND]" not in obs or bad not in obs:
            raise ValueError("Native error observation marker absent")
        injected=[
            {"role":"assistant","content":"","tool_calls":[{
                "id":"v9_failed_read_"+suffix,"type":"function",
                "function":{"name":"read","arguments":{"source":bad}},
            }]},
            {"role":"tool","tool_call_id":"v9_failed_read_"+suffix,
             "name":"read","content":obs},
        ]
        result=dict(row)
        result["messages"]=messages[:idx]+injected+messages[idx:]
        result["training_action_indices"]=[i if i<idx else i+2 for i in selected]
        verify=dict(row.get("verification") or {})
        if verify.get("zero_tool_errors") is not True:
            raise ValueError("Original gold-derived read sequence was not verified clean")
        verify["zero_tool_errors"]=False
        verify["expected_tool_failures_only"]=True
        verify["expected_tool_error_codes"]=["NOT_FOUND"]
        result["verification"]=verify
        metrics=dict(row.get("metrics") or {})
        metrics["tool_calls"]=int(metrics.get("tool_calls") or 0)+1
        metrics["tool_errors"]=int(metrics.get("tool_errors") or 0)+1
        result["metrics"]=metrics
        result["error_conditioned_augmentation"]={
            "protocol":"v9-mu-native-read-not-found-v1",
            "failure_physically_replayed":True,
            "source_read_success_physically_replayed":True,
            "source_link_visible_before_original_action":True,
            "not_observed_agent_behavior":True,
            "supervised_targets_unchanged":True,
        }
        return result
    return None


def build(train: Path, holdout: Path, manifest: Path, musique: Path) -> tuple[list[dict],dict]:
    if sha256(train.read_bytes()).hexdigest()!=TRAIN_SHA:
        raise ValueError("V8 training source SHA mismatch")
    if sha256(musique.read_bytes()).hexdigest()!=MUSIQUE_TRAIN_SHA:
        raise ValueError("Immutable MuSiQue TRAIN source SHA mismatch")
    meta=json.loads(manifest.read_text(encoding="utf8"))
    if (meta.get("train_sha256")!=TRAIN_SHA or meta.get("train_parents")!=304
        or meta.get("train_actions")!=798 or meta.get("no_gaia_training") is not True
        or sha256(holdout.read_bytes()).hexdigest()!=meta.get("validation_sha256")):
        raise ValueError("Original v8 training manifest/holdout mismatch")
    rows=_jsonl(train)
    held=_jsonl(holdout)
    if len(rows)!=N_PARENTS or len(held)!=35:
        raise ValueError("Unexpected SFT TRAIN/holdout parent counts")
    ids={r["task_id"] for r in held}
    if len(ids)!=35 or any(r["task_id"] in ids for r in rows):
        raise ValueError("Parent identity leaked into heldout")

    # Key by original source ID *hash*. Raw MuSiQue human annotations are used
    # ONLY to replay the existing workspace files, not to decide what to read.
    frozen={}
    for raw in _jsonl(musique):
        rid=str(raw.get("id") or "")
        if rid:
            digest=sha256(rid.encode()).hexdigest()
            if digest in frozen:
                raise ValueError("Duplicate original MuSiQue source ID")
            frozen[digest]=raw
    selected_sources=Counter()
    skipped=Counter()
    out=[]
    for row in rows:
        if "gaia" in str(row.get("source") or "").casefold():
            raise ValueError("Forbidden GAIA in original training source")
        if row.get("source")!="musique-train":
            out.append(row)
            continue
        source_id=row.get("source_id_sha256")
        if not isinstance(source_id,str) or source_id not in frozen:
            raise ValueError("MuSiQue parent not found in pinned raw TRAIN source")
        source=frozen[source_id]
        files=_files_from_source(source)
        candidate=augment(row,files)
        if candidate is None:
            skipped["not_source_visible_selected_link"]+=1
            out.append(row)
        else:
            selected_sources["musique-train"]+=1
            out.append(candidate)
    num=sum(selected_sources.values())
    if num<32:
        raise ValueError(f"Too few fully replayable real-source error-context substitutions: {num}; skipped={dict(skipped)}")
    if [r["task_id"] for r in rows]!=[r["task_id"] for r in out]:
        raise ValueError("Task sequence identity drift")
    if sum(len(r["training_action_indices"]) for r in out)!=N_ACTIONS:
        raise ValueError("Supervised action count drift")
    for old,new in zip(rows,out):
        old_actions=[old["messages"][i] for i in old["training_action_indices"]]
        new_actions=[new["messages"][i] for i in new["training_action_indices"]]
        if (old_actions!=new_actions or old.get("expected_answer")!=new.get("expected_answer")
            or old.get("tools")!=new.get("tools")):
            raise ValueError("Supervised target, answer, or tools were changed")
    return out,{
        "protocol":"v9-read-failure-conditioned-source-grounded-qlora-corpus-v1",
        "original_train_sha256":TRAIN_SHA,"musique_train_sha256":MUSIQUE_TRAIN_SHA,
        "original_parents":304,"candidate_parents":304,
        "original_supervised_actions":798,"candidate_supervised_actions":798,
        "original_task_order_unchanged":True,"original_action_labels_unchanged":True,
        "original_holdout_sha256":meta["validation_sha256"],
        "injected_error_context_parents":num,
        "injected_source_counts":dict(sorted(selected_sources.items())),
        "excluded_unverifiable":dict(sorted(skipped.items())),
        "no_training_or_gaia_gold":True,
        "not_observed_agent_recovery":True,
        "limitations":[
            "Injected file typos are synthetic failures on real MuSiQue TRAIN workspaces, not naturally occurring agent errors.",
            "Only originally source-visible, v8-preselected next read actions are eligible; hidden gold support choices are never added.",
            "A separate controlled QLoRA run and novel-source task-level evaluation are needed to claim generalization.",
        ],
    }


def main():
    p=argparse.ArgumentParser()
    for arg in ("train","holdout","manifest","musique","output","report"):
        p.add_argument("--"+arg,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or a.report.exists():
        raise FileExistsError("No overwrite of existing SFT material")
    data,proof=build(a.train,a.holdout,a.manifest,a.musique)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in data))
    proof["candidate_train_sha256"]=sha256(a.output.read_bytes()).hexdigest()
    a.report.write_text(json.dumps(proof,sort_keys=True,indent=2)+"\n")
    print("V9_VERIFIED_MUSIQUE_RECOVERY_CORPUS",json.dumps(proof,sort_keys=True))


if __name__=="__main__":
    main()
