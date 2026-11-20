"""CPU-only real AppWorld TRAIN public API documentation presentation census.

Eight frozen scene-distinct TRAIN tasks; never emits task IDs, prompts, API docs,
parameter names, values, private app names, protected reference or solutions.
No model/token/adapter inference. Only aggregate count of PUBLIC doc fields
available upstream but omitted from the AppWorldDocsTool rendering.
"""
from __future__ import annotations
from collections import Counter
import argparse
from pathlib import Path
import json

from gaia_small_agent.tools.appworld_public_contract_audit import public_doc_visibility_audit
from scripts.appworld_train_first_feedback_fork import (
    task_roster,digest,UPSTREAM_APPWORLD_REV,MODEL_GGUF_SHA,
)

PROTOCOL="APPWORLD-TRAIN8-PUBLIC-DOC-VISIBILITY-SOURCE-AUDIT-V1"


def run():
    from appworld import AppWorld
    roster=task_roster()
    assert len(roster)==8 and len({t.split("_",1)[0] for t in roster})==8
    aggregate=Counter()
    per_scene=[]
    for index,tid in enumerate(roster):
        with AppWorld(task_id=tid, experiment_name=f"public-doc-audit-{index}",
                      ground_truth_mode="full",timeout_seconds=180) as world:
            current=public_doc_visibility_audit(world)
        aggregate.update(current)
        # Only per-scene numeric data, no ID/Task API name/parameter/value/prompt.
        per_scene.append({"scene_index":index,
                          "public_api_count":current.get("public_apis",0),
                          "documented_parameter_count":current.get("public_parameters",0),
                          "excess_parameter_count":current.get(
                              "parameters_hidden_by_first24_cap",0),
                          "default_fields_omitted":current.get(
                              "parameter_default_field_omitted",0),
                          "constraint_fields_omitted":current.get(
                              "parameter_constraints_field_omitted",0),
                          "non_null_defaults_omitted":current.get(
                              "non_null_default_values_omitted",0),
                          "nonempty_constraints_omitted":current.get(
                              "nonempty_public_constraints_omitted",0),
                          "response_examples_omitted":current.get(
                              "public_response_schema_examples_omitted",0)})
    return {
        "protocol":PROTOCOL,
        "status":"REAL_PINNED_UPSTREAM_PUBLIC_DOC_SOURCE_COMPLETE",
        "source_scene_count":len(roster),
        "source_scene_selection_hash":digest(roster),
        "upstream_appworld_sha":UPSTREAM_APPWORLD_REV,
        "qwen_model_sha_frozen_but_not_loaded":MODEL_GGUF_SHA,
        "aggregate_public_only_counts":dict(sorted(aggregate.items())),
        # Eight scene rows may share exactly the same doc counts. Never cite the
        # aggregate as eight independent API documentation catalogs.
        "unique_per_scene_count_profiles":len({digest(s) for s in per_scene}),
        "representative_one_scene_counts":per_scene[0],
        "scene_index_only_counts":per_scene,
        "no_Qwen_inference":True,
        "no_private_task_instruction_output":True,
        "no_api_parameter_names_or_values_output":True,
        "no_gold_or_evaluator_access":True,
        "no_adapter_training":True,
        "no_model_observations_modified":True,
        "accuracy_not_measured":True,
        "claim_boundary":"Visible upstream documentation losses do not prove they cause agent task failure.",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():
        raise FileExistsError("immutable source audit output cannot be overwritten")
    try:
        result=run()
    except Exception as exc:
        print("APPWORLD_PUBLIC_DOC_SOURCE_FAIL_CLOSED",type(exc).__name__)
        raise SystemExit(1) from None
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("REAL_PUBLIC_DOC_FIELDS_ONLY",json.dumps({
        "scenes":result["source_scene_count"],
        "public_apis":result["aggregate_public_only_counts"].get("public_apis",0),
        "public_parameters":result["aggregate_public_only_counts"].get("public_parameters",0),
        "hidden_after_first24":result["aggregate_public_only_counts"].get(
            "parameters_hidden_by_first24_cap",0),
        "default_fields_not_shown":result["aggregate_public_only_counts"].get(
            "parameter_default_field_omitted",0),
        "constraint_fields_not_shown":result["aggregate_public_only_counts"].get(
            "parameter_constraints_field_omitted",0),
        "meaningful_non_null_defaults":result["representative_one_scene_counts"].get(
            "non_null_defaults_omitted",0),
        "meaningful_nonempty_constraints":result["representative_one_scene_counts"].get(
            "nonempty_constraints_omitted",0),
        "public_response_examples_omitted":result["representative_one_scene_counts"].get(
            "response_examples_omitted",0),
        "distinct_count_profiles":result["unique_per_scene_count_profiles"],
        "model_calls":0,
    },sort_keys=True))


if __name__=="__main__":
    main()
