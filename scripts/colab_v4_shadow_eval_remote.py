"""GPU-only, source-disjoint matched Qwen3.5-9B Base vs *trained* v4 GAIA Shadow25.

Uses the existing frozen GAIA shadow partition (not Evaluation100) and fixed
backend/runtime for both arms. No model training or prompt selection here.
Output artifact contains only privacy-safe metrics, never GAIA questions/answers
or model answers. The Colab VM must be released by the invoking workflow.
"""
from __future__ import annotations

from hashlib import sha256
import gc
import itertools
import json
from pathlib import Path
import sys
import zipfile

MODEL = "Qwen/Qwen3.5-9B"
REPO = Path("/content/small-agent-qlora")
ADAPTER_ZIP = Path("/content/qlora-full-v4-adapter.zip")
ADAPTER_DIR = Path("/content/qlora-shadow-v4-adapter")
TOKEN_FILE = Path("/content/gaia-hf-token")
OUT_DIR = Path("/content/gaia-shadow-v4-evidence")


def assert_safe_adapter(zip_path: Path, destination: Path) -> dict:
    if not zip_path.is_file() or zip_path.stat().st_size < 1000000:
        raise ValueError("missing genuine trained v4 adapter zip")
    if destination.exists():
        raise FileExistsError("do not overwrite adapter directory")
    with zipfile.ZipFile(zip_path) as archive:
        entries = archive.namelist()
        if not entries or any(
            name.startswith("/") or ".." in Path(name).parts or "\\" in name
            for name in entries
        ):
            raise ValueError("unsafe adapter archive members")
        required = {"adapter_model.safetensors", "adapter_config.json",
                    "verified-adapter-report.json", "run_config.json"}
        if not required.issubset(set(entries)):
            raise ValueError("not a verified full v4 adapter archive")
        archive.extractall(destination)
    report=json.loads((destination/"verified-adapter-report.json").read_text())
    config=json.loads((destination/"run_config.json").read_text())
    weights=destination/"adapter_model.safetensors"
    measured=sha256(weights.read_bytes()).hexdigest()
    if (report.get("adapter_sha256") != measured
        or report.get("model") != MODEL
        or report.get("optimizer_steps") != 400
        or config.get("training_examples") != 3200
        or config.get("max_steps") != 400
        or config.get("lora_target_modules") != "all-linear"
        or weights.stat().st_size < 1000000):
        raise ValueError("full 400-step adapter identity/integrity contract failed")
    return {"model":MODEL, "adapter_sha256":measured,
            "optimizer_steps":400,"train_actions":3200,
            "source_sha":report.get("source_sha")}


def paired_summary(base: dict, adapter: dict, *, adapter_meta: dict) -> dict:
    from gaia_small_agent.benchmark.gaia100 import summarize_results
    a={row["task_id"]:row for row in base["rows"]}
    b={row["task_id"]:row for row in adapter["rows"]}
    if len(a) != 25 or len(b) != 25 or set(a) != set(b):
        raise ValueError("GAIA shadow benchmark identity mismatch")
    gains=sum(not a[k]["correct"] and b[k]["correct"] for k in a)
    losses=sum(a[k]["correct"] and not b[k]["correct"] for k in a)
    return {
        "schema":"gaia-shadow-base-v4-paired/v1",
        "benchmark":"GAIA 2023 validation Shadow25 (disjoint from original Evaluation100 and Diagnostic25)",
        "same_model": MODEL, "same_runtime":True,
        "max_steps":12,"max_new_tokens":256,"temperature":0,"thinking":False,
        "adapter":adapter_meta,
        "base":base["summary"],"v4":adapter["summary"],
        "wrong_to_correct":gains,"correct_to_wrong":losses,
        "net_gain":gains-losses,
        "paired_only_no_answers":[
            {"task_sha256":sha256(k.encode()).hexdigest(),
             "base_correct":bool(a[k]["correct"]),"v4_correct":bool(b[k]["correct"]),
             "base_completed":bool(a[k]["completed"]),"v4_completed":bool(b[k]["completed"]),
             "base_tool_calls":int(a[k]["tool_calls"]),"v4_tool_calls":int(b[k]["tool_calls"])}
            for k in sorted(a)
        ],
        "limits":"25 GAIA shadow tasks, local tool availability, T4 Transformers rather than previous GGUF llama.cpp; not official leaderboard",
    }


def main() -> None:
    import torch
    device=torch.cuda.get_device_name(0)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    capability=torch.cuda.get_device_capability(0)
    # Native BF16 is unsupported on Tesla T4 (SM 7.5). PyTorch can report
    # emulation as supported, which is unsuitable for 4-bit mixed precision.
    if capability[0] < 8:
        torch.cuda.is_bf16_supported = lambda *args, **kwargs: False
    print("SHADOW_GPU",json.dumps({"device":device, "capability":capability}),flush=True)

    import subprocess
    if REPO.exists():
        raise FileExistsError("unverified prior checkout found on Colab")
    subprocess.run(
        ["git","clone","--depth","100","--branch",
         "fix/qlora-v4-t4-4bit-memory-20261112",
         "https://github.com/kolihhan/small-agent-qlora.git",str(REPO)],
        check=True)
    pinned_sha="144c0a3ef6df7089cdafa3ae8ca17b375804f542"
    subprocess.run(["git","-C",str(REPO),"checkout","--detach",pinned_sha],check=True)
    actual=subprocess.check_output(
        ["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()
    if actual != pinned_sha:
        raise ValueError("evaluation runtime source commit changed")
    subprocess.run(
        [sys.executable,"-m","pip","install","-q",
         "-e",str(REPO)+"[train,eval,files,search]","jinja2>=3.1"],
        check=True)
    # Colab's already-live kernel does not reload editable-install .pth files.
    sys.path[:0]=[str(REPO),str(REPO/"src")]
    from gaia_small_agent.agent.loop import AgentRuntime
    from gaia_small_agent.benchmark.gaia100 import (
        GAIA_REVISION,LOCAL_PROTOCOL_SEED,load_validation,select_gaia_partition)
    from gaia_small_agent.benchmark.runner import run_gaia100
    from gaia_small_agent.model.transformers_qwen import TransformersQwenModel
    from gaia_small_agent.tools import default_tools
    from peft import PeftModel

    adapter_meta=assert_safe_adapter(ADAPTER_ZIP,ADAPTER_DIR)
    if not TOKEN_FILE.is_file():
        raise FileNotFoundError("benchmark token transfer missing")
    hf_token=TOKEN_FILE.read_text().strip()
    TOKEN_FILE.unlink()
    if not hf_token:
        raise ValueError("empty GAIA authorization token")
    dataset=load_validation(token=hf_token,revision=GAIA_REVISION)
    del hf_token
    selected=select_gaia_partition(dataset,partition="shadow",seed=LOCAL_PROTOCOL_SEED)
    if len(selected)!=25 or len({row["task_id"] for row in selected})!=25:
        raise ValueError("GAIA Shadow25 identity invalid")
    OUT_DIR.mkdir(parents=True,exist_ok=False)

    model=TransformersQwenModel(
        model_name=MODEL,quantize_4bit=True,adapter=None,
        enable_thinking=False,max_new_tokens=256,cache_implementation=None,
    )
    model._ids=itertools.count(1)
    arms=[]
    for arm in ("base","v4"):
        if arm=="v4":
            model.model=PeftModel.from_pretrained(model.model,str(ADAPTER_DIR),is_trainable=False)
            model.model.eval()
            model._ids=itertools.count(1)
        path=OUT_DIR/arm
        summary=run_gaia100(
            dataset,lambda:AgentRuntime(model,default_tools(),max_steps=12),
            path,seed=LOCAL_PROTOCOL_SEED,dataset_revision=GAIA_REVISION,
            partition="shadow",
            run_config={
                "arm":arm,"model":MODEL,"adapter_sha256":adapter_meta["adapter_sha256"] if arm=="v4" else None,
                "source":"GAIA 2023 validation Shadow25",
                "max_steps":12,"max_new_tokens":256,
                "decoding":"deterministic","native_bf16":capability[0]>=8,
            },
        )
        records=[json.loads(line) for line in (path/"results.jsonl").read_text().splitlines() if line.strip()]
        if len(records)!=25 or summary["total"]!=25:
            raise ValueError(f"{arm}: incomplete GAIA shadow evidence")
        arms.append({"summary":summary,"rows":records})
        print("SHADOW_ARM_FINAL",json.dumps({
            "arm":arm,"correct":summary["correct"],"completion_rate":summary["completion_rate"],
            "tool_calls":summary["tool_calls"],
        },sort_keys=True),flush=True)

    report=paired_summary(arms[0],arms[1],adapter_meta=adapter_meta)
    (OUT_DIR/"paired-private-safe.json").write_text(
        json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("SHADOW_PAIRED_FINAL",json.dumps({
        k:v for k,v in report.items() if k not in {"paired_only_no_answers"}
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
