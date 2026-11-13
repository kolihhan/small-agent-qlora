"""Pure CPU contracts for GAIA Shadow25 matched v4 evaluation output."""
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from scripts.colab_v4_200step_shadow_eval_remote import assert_safe_adapter,paired_summary


def _rows(*, treated: bool):
    rows=[]
    for i in range(25):
        correct=(i<10) if not treated else (i<11 and i != 1)
        rows.append({
            "task_id":f"id-{i}","correct":correct,
            "completed":i<20,"tool_calls":2,
        })
    return {"rows":rows,"summary":{
        "total":25,"correct":sum(r["correct"] for r in rows),
        "completion_rate":20/25,"tool_calls":50,
    }}


def test_shadow_paired_scores_are_validated_on_identical_task_identity():
    m={"adapter_sha256":"sha256","optimizer_steps":200}
    r=paired_summary(_rows(treated=False),_rows(treated=True),adapter_meta=m)
    assert r["base"]["correct"]==10
    assert r["v4"]["correct"]==10
    assert r["wrong_to_correct"]==1 and r["correct_to_wrong"]==1
    assert r["net_gain"]==0
    assert len(r["paired_only_no_answers"])==25
    assert all("model_answer" not in x for x in r["paired_only_no_answers"])
    assert all(len(x["task_sha256"])==64 for x in r["paired_only_no_answers"])


def test_shadow_identity_mismatch_rejected_even_with_matched_cardinality():
    base=_rows(treated=False)
    candidate=_rows(treated=True)
    candidate["rows"][0]["task_id"]="different"
    with pytest.raises(ValueError, match="identity mismatch"):
        paired_summary(base,candidate,adapter_meta={})


def test_full_trained_adapter_integrity_cannot_be_bypassed(tmp_path):
    output=tmp_path/"out"
    archive=tmp_path/"full.zip"
    blob=b"genuine-fake-weights" * 100000
    report={"adapter_sha256":hashlib.sha256(blob).hexdigest(),
            "model":"Qwen/Qwen3.5-9B","optimizer_steps":200,
            "source_sha":"frozen"}
    cfg={"training_examples":3200,"max_steps":200,
         "lora_target_modules":"all-linear"}
    with zipfile.ZipFile(archive,"w") as z:
        z.writestr("adapter_model.safetensors",blob)
        z.writestr("adapter_config.json",json.dumps({"peft_type":"LORA"}))
        z.writestr("verified-adapter-report.json",json.dumps(report))
        z.writestr("run_config.json",json.dumps(cfg))
    meta=assert_safe_adapter(archive,output)
    assert meta["adapter_sha256"]==report["adapter_sha256"]
    assert meta["optimizer_steps"]==200
    with pytest.raises(FileExistsError):
        assert_safe_adapter(archive,output)
    bad=tmp_path/"malicious.zip"
    with zipfile.ZipFile(bad,"w") as z:
        z.writestr("../x","unsafe")
        # Exercise the traversal guard AFTER the independent minimum-size gate.
        z.writestr("padding.bin",b"x"*1048576)
    with pytest.raises(ValueError,match="unsafe"):
        assert_safe_adapter(bad,tmp_path/"bad")


def test_wrong_or_smoke_adapter_rejected(tmp_path):
    archive=tmp_path/"smoke.zip"
    with zipfile.ZipFile(archive,"w") as z:
        z.writestr("adapter_model.safetensors",b"x" * 1048576)
        z.writestr("adapter_config.json","{}")
        z.writestr("verified-adapter-report.json",json.dumps({
            "model":"Qwen/Qwen3.5-9B","optimizer_steps":2,
            "adapter_sha256":hashlib.sha256(b"x"*1048576).hexdigest()
        }))
        z.writestr("run_config.json",json.dumps({
            "training_examples":3200,"max_steps":2,"lora_target_modules":"all-linear"
        }))
    with pytest.raises(ValueError,match="200-step"):
        assert_safe_adapter(archive,tmp_path/"unverified")
