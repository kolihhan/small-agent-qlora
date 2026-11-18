"""Execute 8 real optimizer steps of original Qwen3.5-9B QLoRA in a Colab GPU.

Called only by the GitHub Actions official google-colab-cli workflow, using
the user's existing COLAB_ADC_JSON GitHub Action secret. Immutable 40 native-source
TRAIN next-actions; full 4096 tokens; zero GAIA/VAL supervision. A physical
checkpoint must exist before any successful training claim.
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

SOURCE_COMMIT="__SOURCE_COMMIT__"
TRAINER_COMMIT="6465698908c6883b0f91451785caa11ebfadb090"
MODEL="Qwen/Qwen3.5-9B"
MODEL_REV="c202236235762e1c871ad0ccb60c8ee5ba337b9a"
DATA_SHA="c900850f676834a5db295ae6a9b9f5b1ccbf5956dc78fdfef2f18dcb560c3835"
DATA=Path("/content/native-qwen9b-train40.jsonl")
OUTPUT=Path("/content/native40-9b-real-gpu8")
ARCHIVE=Path("/content/native40-9b-gpu8-checkpoint.zip")
REPORT=Path("/content/native40-9b-gpu8-report.json")
STEPS=8
CONTEXT=4096
PROTOCOL="qwen35-9b-native-tools-40-8step-colab-v1"


def command(*args):
    print("NATIVE40_REMOTE_STEP",str(args[0]),flush=True)
    subprocess.run([str(x) for x in args],check=True)


def data_contract():
    if len(SOURCE_COMMIT)!=40 or SOURCE_COMMIT=="__SOURCE_COMMIT__":
        raise RuntimeError("source code revision not frozen to GitHub SHA")
    if not DATA.is_file() or sha256(DATA.read_bytes()).hexdigest()!=DATA_SHA:
        raise RuntimeError("original native40 TRAIN actions digest mismatch")
    rows=[json.loads(s) for s in DATA.read_text().splitlines() if s.strip()]
    if len(rows)!=40 or len({r["task_id"] for r in rows})!=40:
        raise ValueError("source 48 distinct target decisions not present")
    states=Counter(r["capability"] for r in rows)
    if states!={"native_missing_evidence_continue":20,"native_observed_evidence_finalize":20}:
        raise ValueError("20/20 native evidence-conditioned source balance changed")
    parents={r.get("task_id") for r in rows}
    if len(parents)!=40 or None in parents:
        raise ValueError("original TRAIN parent-disjoint exposure mismatch")
    for r in rows:
        if (r.get("source") not in ("2wikimultihopqa-train", "hotpotqa-distractor-train")
            or r.get("verified") is not True
            or r.get("source_native_actions_unchanged") is not True
            or not isinstance(r.get("original_observations_before_action"), int)
            or len(r.get("training_action_indices",[]))!=1
            or any(x in str(r["task_id"]).casefold() for x in ("gaia","validation"))):
            raise ValueError("source not permitted / original action not grounded")
    print("NATIVE40_SOURCE_IMMUTABLE_48",json.dumps({
        "data_sha256":DATA_SHA,"source_parent_tasks":len(parents),
        "original_source_next_actions":40,"states":dict(states),
        "gaia_supervision":False
    },sort_keys=True),flush=True)


def pack_checkpoint(kind:str, *,error:str|None=None):
    if not OUTPUT.exists():
        return
    available=sorted(OUTPUT.glob("checkpoint-*"))
    has_final=(OUTPUT/"adapter_model.safetensors").is_file()
    if not available and not has_final:
        return
    # Recovered partial checkpoint is not a completed adapter.
    status={
        "protocol":PROTOCOL,
        "kind":kind,
        "error_type":error,
        "source_commit":SOURCE_COMMIT,
        "trainer_commit":TRAINER_COMMIT,
        "model_revision":MODEL_REV,
        "original_data_sha256":DATA_SHA,
        "full_context_tokens":CONTEXT,
        "target_optimizer_steps":STEPS,
        "checkpoint_directories":[p.name for p in available],
        "final_adapter_exists":has_final,
        "gaia_checked":False,
        "validated_external_tasks":False,
        "promoted":False
    }
    REPORT.write_text(json.dumps(status,sort_keys=True,indent=2)+"\n")
    with zipfile.ZipFile(ARCHIVE,"w",compression=zipfile.ZIP_STORED,
                         allowZip64=True) as z:
        for candidate in sorted(OUTPUT.rglob("*")):
            if candidate.is_file():
                z.write(candidate,arcname="trainer/"+str(candidate.relative_to(OUTPUT)))
        z.write(REPORT,arcname="native40-9b-gpu8-report.json")
    print("NATIVE40_COLAB_CHECKPOINT_PACKED",json.dumps({
        "kind":kind,"archive_bytes":ARCHIVE.stat().st_size,
        "checkpoints":[p.name for p in available],"has_final":has_final,
    },sort_keys=True),flush=True)


def main():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("NO_GPU: cannot classify CPU preparation as training")
    prop=torch.cuda.get_device_properties(0)
    vram=prop.total_memory/2**30
    if vram<14.5:
        raise RuntimeError("GPU_TOO_SMALL: T4-class >=14.5GiB required")
    data_contract()
    if any(x.exists() for x in (OUTPUT,ARCHIVE,REPORT)):
        raise RuntimeError("refuse to overwrite earlier Colab adapter workspace")
    trainer=Path("/content/qlora-native40-compatible-t4")
    if trainer.exists():
        raise RuntimeError("trainer workspace already in use")
    command("git","clone","--branch","experiment/qlora-v8-durable-stage35-20261115",
            "--depth","10","https://github.com/kolihhan/small-agent-qlora.git",str(trainer))
    command("git","-C",str(trainer),"checkout","--detach",TRAINER_COMMIT)
    revision=subprocess.check_output(
        ["git","-C",str(trainer),"rev-parse","HEAD"],text=True).strip()
    if revision!=TRAINER_COMMIT:
        raise RuntimeError("historical T4-compatible trainer revision drift")
    command(sys.executable,"-m","pip","install","-q","-e",str(trainer)+"[train]",
            "jinja2>=3.1","sentencepiece>=0.2")
    sys.path.insert(0,str(trainer/"src"))
    from gaia_small_agent.training.qlora import (
        train_qlora,load_verified_rows,_render_turn_examples,
        _tokenize_completion_examples,_configure_trl_legacy_addmm
    )
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download
    import trl.kernels.chunked_logprob as trl_chunked
    capability=torch.cuda.get_device_capability(0)
    if _configure_trl_legacy_addmm(trl_chunked,capability=capability):
        a=torch.ones((2,3),device="cuda",dtype=torch.float16)
        b=torch.ones((3,2),device="cuda",dtype=torch.float16)
        acc=torch.zeros((2,2),device="cuda",dtype=torch.float32)
        trl_chunked._addmm_fp32(acc,a,b)
        torch.cuda.synchronize()
        assert torch.all(acc==3)
        print("NATIVE40_T4_TRL_MIXED_GEMM_PREFLIGHT_PASS",flush=True)
    original=snapshot_download(MODEL,revision=MODEL_REV)
    tokenizer=AutoTokenizer.from_pretrained(original,trust_remote_code=True)
    rows=load_verified_rows(DATA)
    examples=_render_turn_examples(tokenizer,rows)
    tokens=_tokenize_completion_examples(
        tokenizer,examples,max_length=CONTEXT,require_full_prompt=True)
    types=Counter(e["metadata"]["action_type"] for e in examples)
    if len(tokens)!=40 or types!={"tool_call":20,"final":20}:
        raise RuntimeError("original source tool/final supervision drift")
    print("NATIVE40_COLAB_REAL_9B_GPU_PREFLIGHT_PASS",json.dumps({
        "gpu":prop.name,"vram_gib":round(vram,2),
        "source_sha":SOURCE_COMMIT,"trainer_sha":TRAINER_COMMIT,
        "model_revision":MODEL_REV,
        "full_prompt_completion_examples":len(tokens),
        "types":dict(types),"4bit_nf4":True,
        "lora_target_modules":["q_proj","v_proj"],
        "max_steps":STEPS,"gaia_supervision":False
    },sort_keys=True),flush=True)

    start=time.monotonic()
    try:
        train_qlora(
            data_path=DATA,output_dir=OUTPUT,
            model_name=original,max_length=CONTEXT,
            epochs=1.0,learning_rate=5e-5,
            lora_r=8,lora_alpha=16,lora_dropout=0.05,
            max_steps=STEPS,save_steps=4,
            gradient_accumulation_steps=1,
            lora_target_modules="q_proj,v_proj",
            exposure_count=None,require_full_prompt=True,
            preserve_large_half_parameters=True,
        )
        state_path=OUTPUT/"checkpoint-8"/"trainer_state.json"
        required=(
            OUTPUT/"adapter_model.safetensors",
            OUTPUT/"adapter_config.json",
            OUTPUT/"run_config.json",state_path
        )
        if not all(x.is_file() for x in required):
            raise RuntimeError("actual final adapter/trainer checkpoint absent")
        state=json.loads(state_path.read_text())
        run=json.loads((OUTPUT/"run_config.json").read_text())
        if (int(state.get("global_step",-1))!=STEPS
            or run.get("training_examples")!=40
            or run.get("max_length")!=CONTEXT
            or run.get("max_steps")!=STEPS
            or run.get("require_full_prompt") is not True
            or run.get("preserve_large_half_parameters") is not True):
            raise ValueError("physical Qwen9B optimizer-step/context proof failed")
        weights=(OUTPUT/"adapter_model.safetensors").read_bytes()
        if len(weights)<100_000:
            raise RuntimeError("adapter weights empty")
        physical={
            "protocol":PROTOCOL,
            "actual_optimizer_steps":STEPS,
            "source_actions":40,"sources":40,
            "model_revision":MODEL_REV,
            "adapter_sha256":sha256(weights).hexdigest(),
            "adapter_bytes":len(weights),
            "duration_seconds":round(time.monotonic()-start),
            "full_context_tokens":CONTEXT,
            "old_v3b_overwritten":False,
            "independent_qa_validated":False,
            "gaia_score_updated":False,
            "promoted":False
        }
        (OUTPUT/"physical-gpu8-proof.json").write_text(
            json.dumps(physical,indent=2,sort_keys=True)+"\n")
        pack_checkpoint("verified_8_steps")
        print("NATIVE40_QLORA_REAL_GPU8_OPTIMIZER_PROOF",
              json.dumps(physical,sort_keys=True),flush=True)
    except BaseException as exc:
        pack_checkpoint("partial_training_failure",error=type(exc).__name__)
        raise


if __name__=="__main__":
    main()
