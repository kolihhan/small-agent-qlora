"""Matched v7 vs frozen Base on 11 source-disjoint real tool parents.

This is an *actual autoregressive next-action test* (not teacher-forced NLL).
It evaluates exactly the first observed tool action plus last final decision of
each untouched HotpotQA/2Wiki parent, with the SAME prompts/token budget and
the SAME 4-bit base-model instance for Base and v7. A v3B/GAIA score is NOT
claimed by this small independent source holdout.

Executed only inside the trusted GPU Colab VM. The runner pins inputs and the
training artifact SHA, downloads the result and stores full paired examples.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

EVAL_SHA = "__EVAL_SHA__"
TRAIN_SOURCE_SHA = "686e8aca022713d0dbcc4b2b2690251465916c8c"
ADAPTER_SHA256 = "c5033f28a6947b687ada54731e876ad0f87d93164f9a02f516e6323d179eaa85"
BRANCH = "experiment/qlora-v7-helout-nextaction-eval-20261113"
MODEL = "Qwen/Qwen3.5-9B"
HOLDOUT = Path("/content/qlora-v7-heldout.jsonl")
TRAIN = Path("/content/qlora-v7-train.jsonl")
AUDIT = Path("/content/qlora-v7-audit.json")
ADAPTER_ZIP = Path("/content/qlora-v7-adapter.zip")
RESULT = Path("/content/qlora-v7-heldout-eval.json")
ADAPTER_DIR = Path("/content/qlora-v7-eval-adapter")


def sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 11 or len({str(x["task_id"]) for x in rows}) != len(rows):
        raise ValueError("expected exactly eleven unique observed heldout parents")
    allowed = {"hotpotqa-distractor-train", "2wikimultihopqa-train"}
    if {x["source"] for x in rows} != allowed:
        raise ValueError("source-shift holdout provenance mismatch")
    for row in rows:
        if row.get("verified") is not True or not isinstance(row.get("messages"), list):
            raise ValueError("unverified heldout parent")
    return rows


def select_early_tool_and_final(rows: list[dict]) -> set[str]:
    """Freeze selection independent of any model predictions or benchmark labels."""
    chosen = set()
    for row in rows:
        task = str(row["task_id"])
        messages = row["messages"]
        tool = [
            i for i, msg in enumerate(messages)
            if msg.get("role") == "assistant" and msg.get("tool_calls")
        ]
        final = [
            i for i, msg in enumerate(messages)
            if msg.get("role") == "assistant" and not msg.get("tool_calls")
        ]
        if not tool or not final or tool[0] >= final[-1]:
            raise ValueError(f"heldout real parent lacks tool-to-final path: {task}")
        chosen.add(f"{task}:{tool[0]}")
        chosen.add(f"{task}:{final[-1]}")
    if len(chosen) != 22:
        raise ValueError("must select 11 first tool decisions + 11 final decisions")
    return chosen


def main() -> None:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU absent: no CPU fall-through for Qwen9B inference")
    if EVAL_SHA == "__EVAL_SHA__" or len(EVAL_SHA) != 40:
        raise ValueError("eval checkout must be SHA-pinned")
    aud = json.loads(AUDIT.read_text())
    if (aud.get("schema") != "qlora-v7-whole-task-source-balanced-20261113-v1"
        or aud.get("holdout_tasks") != 11 or aud.get("train_actions") != 522):
        raise ValueError("wrong source curriculum")
    for path,key in [(TRAIN,"train_sha256"),(HOLDOUT,"holdout_sha256")]:
        if sha(path) != aud[key]:
            raise ValueError("original training/holdout SHA changed")

    with zipfile.ZipFile(ADAPTER_ZIP) as z:
        names = set(z.namelist())
        required = {"adapter_model.safetensors","adapter_config.json","verified-v7-report.json"}
        if not required <= names:
            raise ValueError("not a complete real adapter")
        provenance = json.loads(z.read("verified-v7-report.json"))
        if (provenance.get("schema") != "qlora-v7-actual-nf4-sft-checkpoint-v1"
            or provenance.get("source_sha") != TRAIN_SOURCE_SHA
            or provenance.get("adapter_sha256") != ADAPTER_SHA256
            or sha256(z.read("adapter_model.safetensors")).hexdigest() != ADAPTER_SHA256
            or provenance.get("train_actions") != 522
            or provenance.get("heldout_parents_unseen") != 11):
            raise ValueError("training adapter provenance mismatch")
        ADAPTER_DIR.mkdir(exist_ok=False)
        for name in required:
            (ADAPTER_DIR/name).write_bytes(z.read(name))
    print("V7_HOLDOUT_ADAPTER_PHYSICALLY_VERIFIED",ADAPTER_SHA256,flush=True)

    rows = load_rows(HOLDOUT)
    ids = select_early_tool_and_final(rows)
    repo = Path("/content/small-agent-qlora-v7-eval")
    subprocess.run(["git","clone","--branch",BRANCH,"--depth","100",
                    "https://github.com/kolihhan/small-agent-qlora.git",str(repo)],check=True)
    subprocess.run(["git","-C",str(repo),"checkout","--detach",EVAL_SHA],check=True)
    checkout = subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],
                                       text=True).strip()
    if checkout != EVAL_SHA:
        raise ValueError("evaluation source checkout mismatch")
    subprocess.run([sys.executable,"-m","pip","install","-q","-e",
                    str(repo)+"[train,dev,files]","jinja2>=3.1"],check=True)
    sys.path[:0] = [str(repo/"src"),str(repo)]
    from gaia_small_agent.model.transformers_qwen import TransformersQwenModel
    from gaia_small_agent.training.next_action_eval import evaluate_next_actions
    from peft import PeftModel

    # Matched deterministic generated decisions, no loss-oracle promotion.
    model = TransformersQwenModel(
        model_name=MODEL, quantize_4bit=True, enable_thinking=False,
        max_new_tokens=160, cache_implementation=None,
    )
    t0=time.time()
    base = evaluate_next_actions(HOLDOUT,lambda:model,example_ids=ids)
    print("V7_HELDOUT_BASE_DONE",json.dumps({
        "total":base["total"],"correct":base["correct"],
        "by_action_type":base["by_action_type"],
        "seconds":round(time.time()-t0)}),flush=True)
    model.model = PeftModel.from_pretrained(
        model.model,str(ADAPTER_DIR),is_trainable=False
    )
    model.model.eval()
    t1=time.time()
    v7 = evaluate_next_actions(HOLDOUT,lambda:model,example_ids=ids)
    print("V7_HELDOUT_V7_DONE",json.dumps({
        "total":v7["total"],"correct":v7["correct"],
        "by_action_type":v7["by_action_type"],
        "seconds":round(time.time()-t1)}),flush=True)

    b={r["example_id"]:r for r in base["examples"]}
    v={r["example_id"]:r for r in v7["examples"]}
    if set(b)!=ids or set(v)!=ids:
        raise ValueError("different model decision cohorts")
    paired=[{
        "example_id":key,"action_type":b[key]["action_type"],
        "base_correct":b[key]["correct"],"v7_correct":v[key]["correct"],
        "base_error":b[key]["error"],"v7_error":v[key]["error"],
        "base_predicted_tool_calls":b[key]["predicted_tool_calls_full"],
        "v7_predicted_tool_calls":v[key]["predicted_tool_calls_full"],
        "base_predicted_final":b[key]["predicted_final"][:600],
        "v7_predicted_final":v[key]["predicted_final"][:600],
    } for key in sorted(ids)]
    gains=sum(not r["base_correct"] and r["v7_correct"] for r in paired)
    losses=sum(r["base_correct"] and not r["v7_correct"] for r in paired)
    result={
        "schema":"v7-heldout-real-nextaction-paired22/v1",
        "cohort":"11 real HotpotQA / 2Wiki parent traces unseen by training",
        "source_sha256":aud["holdout_sha256"],
        "training_run_id":43418636161,
        "adapter_sha256":ADAPTER_SHA256,
        "generation":"greedy; no thinking; max_new_tokens=160; 4-bit NF4; same base instance",
        "base_total":base["total"],"base_correct":base["correct"],
        "v7_total":v7["total"],"v7_correct":v7["correct"],
        "base_by_action_type":base["by_action_type"],
        "v7_by_action_type":v7["by_action_type"],
        "wrong_to_correct":gains,"correct_to_wrong":losses,
        "net_gain":gains-losses,
        "paired_examples":paired,
        "gaia_accuracy":"NOT_TESTED",
        "promote":False,
        "caveat":"Very small source-disjoint in-domain next-action diagnostic, not "
                 "end-to-end GAIA or closed-loop policy success; tool exact-match "
                 "and final gaia_score on reference prefix.",
    }
    if result["base_correct"]+gains-losses != result["v7_correct"]:
        raise AssertionError("paired totals disagree")
    RESULT.write_text(json.dumps(result,indent=2,ensure_ascii=False,sort_keys=True)+"\n")
    print("V7_HELDOUT_PAIRED_RESULT",json.dumps({
        k:v for k,v in result.items() if k!="paired_examples"
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
