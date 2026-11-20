"""Source-only pre-registration of NEW TRAIN scene-disjoint parity pilot.

No Qwen, no task prompt, no credentials, no evaluator. Selection uses only
public TRAIN IDs and an immutable seed, NEVER task success/401 prevalence.
Only safe task identifier DIGESTS (not raw task text/IDs) leave runner.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from scripts.appworld_train_first_feedback_fork import (
    task_roster, MODEL_GGUF_SHA, UPSTREAM_APPWORLD_REV,
)
from gaia_small_agent.tools.appworld_parity_safe_export import PROTOCOL as SAFE_REPORT_PROTOCOL

PROTOCOL="APPWORLD-PARITY-PREREG-NEW-TRAIN2-SCENES-2REPEATS-MATCHED-V1"
SEED="appworld-official-supervisor-paired-fresh-train2-v1-2026-11-20"
N_SCENES=2
N_REPEATS=2
ARMS=("baseline","parity")
MAX_STEPS=20
MAX_API_CALLS=48
MAX_MODEL_OUTPUT_TOKENS=384
MODEL_TEMPERATURE=0


def _hash(x:str)->str:
    return sha256(x.encode("utf-8")).hexdigest()


def select_task_ids(train_ids:list[str],seen_pilot_ids:list[str])->list[str]:
    """Exclude entire previously exposed TRAIN8 scenes; do not peek at tasks."""
    if not train_ids or len(set(train_ids))!=len(train_ids):
        raise ValueError("invalid TRAIN task roster")
    seen_scenes={x.split("_",1)[0] for x in seen_pilot_ids}
    by_scene={}
    for tid in train_ids:
        scene=tid.split("_",1)[0]
        if scene not in seen_scenes:
            by_scene.setdefault(scene,[]).append(tid)
    if len(by_scene)<N_SCENES:
        raise ValueError("insufficient scene-disjoint source tasks")
    ordered_scenes=sorted(by_scene,key=lambda s:_hash(SEED+":scene:"+s))
    picked=[]
    for scene in ordered_scenes[:N_SCENES]:
        picked.append(min(by_scene[scene],key=lambda t:_hash(SEED+":task:"+t)))
    assert len(picked)==N_SCENES
    assert len({x.split("_",1)[0] for x in picked})==N_SCENES
    assert all(x.split("_",1)[0] not in seen_scenes for x in picked)
    return picked


def freeze(train_ids:list[str],seen_pilot_ids:list[str])->dict:
    chosen=select_task_ids(train_ids,seen_pilot_ids)
    assert len(seen_pilot_ids)==8 and len({x.split("_",1)[0] for x in seen_pilot_ids})==8
    return {
        "protocol":PROTOCOL,
        "preregistered_seed":SEED,
        "source_split":"train",
        "total_available_train_task_ids":len(train_ids),
        "previously_exposed_pilot_scenes_excluded":8,
        "selected_new_scene_count":N_SCENES,
        "selected_task_id_sha256":[_hash(x) for x in chosen],
        "selected_scene_id_sha256":[_hash(x.split("_",1)[0]) for x in chosen],
        "ordered_all_train_ids_sha256":_hash("\n".join(sorted(train_ids))),
        "selected_task_ids_are_not_exposed_in_artifacts":True,
        "new_scenes_disjoint_from_original_train8":True,
        "all_other_prior_research_exposures_unknown":True,
        "n_repeats_per_arm_per_scene":N_REPEATS,
        "arms":list(ARMS),
        "max_agent_steps":MAX_STEPS,
        "max_model_new_tokens_each_turn":MAX_MODEL_OUTPUT_TOKENS,
        "max_actual_api_calls":MAX_API_CALLS,
        "model_sampling_temperature":MODEL_TEMPERATURE,
        "pinned_model_SHA256":MODEL_GGUF_SHA,
        "pinned_appworld_commit":UPSTREAM_APPWORLD_REV,
        "tool_change_is_only_off_vs_on_parity":True,
        "frozen_same_task_and_fresh_world_per_arm_repeat":True,
        "baseline_and_parity_first_model_inputs_are_intentionally_different":True,
        "official_task_success_primary_endpoint":True,
        "nested_state_tests_secondary":True,
        "sup_auth_login_counts_diagnostic_not_primary":True,
        "paired_rescues_and_regressions_reported":True,
        "repeat_instability_must_be_reported":True,
        "safe_report_protocol":SAFE_REPORT_PROTOCOL,
        "no_Qwen_training_or_execution_in_this_freeze":True,
        "not_a_promotion_holdout":True,
        "no_task_text_credential_values_or_credential_hashes_exported":True,
    }


def require_frozen(meta:dict,train_ids:list[str],seen_pilot_ids:list[str])->list[str]:
    current=freeze(train_ids,seen_pilot_ids)
    if meta!=current:
        raise ValueError("source/budget/model/selection cohort drift")
    chosen=select_task_ids(train_ids,seen_pilot_ids)
    assert [_hash(x) for x in chosen]==meta["selected_task_id_sha256"]
    return chosen


def main():
    import argparse
    a=argparse.ArgumentParser()
    a.add_argument("--out",required=True,type=Path)
    p=a.parse_args()
    from appworld import load_task_ids
    original=task_roster()
    meta=freeze(load_task_ids("train"),original)
    if p.out.exists():raise FileExistsError("preregistration immutable")
    p.out.parent.mkdir(parents=True,exist_ok=True)
    p.out.write_text(json.dumps(meta,sort_keys=True,indent=2)+"\n")
    print("NEW_TRAIN_SCENES_SOURCE_ONLY_PREREG",json.dumps({
        "selected_scene_count":meta["selected_new_scene_count"],
        "previous_pilot_scene_overlap":0,
        "arms":meta["arms"],"repeats":meta["n_repeats_per_arm_per_scene"],
        "model_inference":False,
    },sort_keys=True))


if __name__=="__main__":main()
