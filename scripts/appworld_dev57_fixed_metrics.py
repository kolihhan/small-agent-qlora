"""Permanent AppWorld DEV57 main agent benchmark, stage-aware CPU scoring.

The DEV roster is frozen by the original official task-ID SHA, NOT student
outcomes or reference answers. The 8 previously probed DEV tasks are marked
as historical exposure, not silently advertised as untouched. No DEV gold
enters the model, trace stages or selector; AppWorld reference request count
can only be added as a post-run diagnostic difficulty bucket.

AppWorld has only 57 DEV tasks / 19 scenarios: sampling K trajectories from
a task NEVER creates more than 57 independent task outcomes. Do not assert
5-percentage-point statistical power at this sample size without a predeclared
power calculation. Never swap AppWorld/Hotpot/GAIA primary mid-experiment.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from hashlib import sha256
import json

PROTOCOL="SMALL-AGENT-APPWORLD-DEV57-FIXED-MAIN-STAGE-V1"
UPSTREAM_COMMIT="42b5bcf3cd334fee33f0c37c02070a9f5807add5"
MODEL_GGUF_SHA="8c4cc24f54473f3d4e080d4ec7fd2a40d0bbec0f94f3c807cd62694905357d54"
OFFICIAL_DEV_IDS_SHA="d36670b383552f4c586baf1f84f0d53fcd3e05db5698b1532b2291ace23288a8"
N=57
SCENARIOS=19
EXPOSED_PILOTS=8
PRIOR_SEED="appworld-dev4-taskid-source-only-pinned-20261117"
SECOND_SEED="appworld-new4-original-qwen-baseline-vs-http-error-observation-20261117"
PRIOR4_SHA="12d0a62d37b35ceb8fa5b9d8ebad2821712e28be34d14ecdca0d8240358b0e9c"
NEXT4_SHA="39bc7a01d2ee011958bd36d2ea2bf3c27e709600f065b372a16ec8fa04a437de"
STAGES=("tool_call_validity","api_http_failure","post_error_recovery",
        "final_rate","official_state_success")
DIFFICULTY_BUCKETS=("0-5","6-11","12+")


def original_dev_ids_sha(ids: list[str]) -> str:
    if not ids or len(ids)!=len(set(ids)):
        raise ValueError("duplicate or empty DEV task identities")
    return sha256("\n".join(sorted(ids)).encode()).hexdigest()


def candidate4(ids,seed,used_scenarios=None):
    seen=set(used_scenarios or ())
    chosen=[]
    for tid in sorted(ids,key=lambda t:sha256(f"{seed}:{t}".encode()).hexdigest()):
        group=tid.split("_",1)[0]
        if group in seen:continue
        seen.add(group)
        chosen.append(tid)
        if len(chosen)==4:break
    if len(chosen)!=4:raise ValueError("not enough prior scenario-disjoint tasks")
    return chosen


def previous_eight(ids):
    old=candidate4(ids,PRIOR_SEED)
    # Old first-four Qwen experiment sorted task IDs before hashing its SHA.
    expected=sha256(json.dumps(old,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    if expected!=PRIOR4_SHA:
        raise ValueError("previous original DEV4 source set changed")
    new=candidate4(ids,SECOND_SEED,{t.split("_",1)[0] for t in old})
    newer=sha256(json.dumps(new,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    if newer!=NEXT4_SHA:
        raise ValueError("previous second DEV4 source set changed")
    if len(set(old+new))!=8:return []
    return old+new


def freeze_official_dev(ids: list[str]) -> dict:
    if (len(ids)!=N or original_dev_ids_sha(ids)!=OFFICIAL_DEV_IDS_SHA or
        len({t.split("_",1)[0] for t in ids})!=SCENARIOS):
        raise ValueError("original pinned AppWorld DEV57 task identity/scenario drift")
    seen=previous_eight(ids)
    if len(seen)!=EXPOSED_PILOTS:
        raise ValueError("must identify all eight previously exposed pilot tasks")
    hashes=sorted(sha256(t.encode()).hexdigest() for t in ids)
    seen_hash=sorted(sha256(t.encode()).hexdigest() for t in seen)
    return {
      "status":"APPWORLD_MAIN_DEV57_FROZEN_BEFORE_NEW_MODEL_RUN",
      "protocol":PROTOCOL,"split":"dev","n":N,"scenario_count":SCENARIOS,
      "upstream_commit":UPSTREAM_COMMIT,"model_sha256":MODEL_GGUF_SHA,
      "task_ids_sha256":OFFICIAL_DEV_IDS_SHA,
      "task_hashes_sha256":sha256("\n".join(hashes).encode()).hexdigest(),
      "previously_exposed_task_hashes_sha256":sha256("\n".join(seen_hash).encode()).hexdigest(),
      "previously_exposed":EXPOSED_PILOTS,
      "estimated_unexposed_in_roster":N-EXPOSED_PILOTS,
      "source_selected_without_student_outputs":True,
      "reference_api_calls_not_used_for_population_selection":True,
      "difficulty_labels_only_post_run":True,
      "main_benchmark_fixed":True,
      "no_training":True,
      "claim_boundary":"All 57 original AppWorld DEV tasks, 19 scenes; eight prior physically probed tasks flagged. Full success and stage metrics only after 57 complete; reference trajectory not in model, never treat 57×K as 57K independent tasks."
    }


def extract_stage_record(task_id: str, result, *, api_calls: int,
                         real_http_failures: int, official_state_success: bool,
                         state_tests_pass: int, state_tests_total: int,
                         observed_reference_call_count: int | None = None) -> dict:
    """Convert in-memory protected trace to non-sensitive counts ONLY."""
    if min(api_calls,real_http_failures,state_tests_pass,state_tests_total)<0:
        raise ValueError("negative physical metric")
    if real_http_failures>api_calls or state_tests_pass>state_tests_total:
        raise ValueError("impossible official score or API error count")
    trace=result.trace
    actual_tools=[t for t in trace if t.kind=="tool_call"]
    tool_results=[t for t in trace if t.kind=="tool_result"]
    # Count error-recovery as any subsequent appworld_call tool result with
    # ok=True; an operational proxy, NOT proof it answered the user task.
    http_error_indices=[i for i,e in enumerate(trace) if e.kind=="tool_result"
                       and e.data.get("error_code")=="HTTP_ERROR"]
    recovered=0
    for index in http_error_indices:
        if any(e.kind=="tool_result" and e.data.get("name")=="appworld_call"
               and e.data.get("ok") is True for e in trace[index+1:]):
            recovered+=1
    valid_form=max(0,len(actual_tools)-int(result.metrics.invalid_tool_calls))
    if observed_reference_call_count is None:
        difficulty="unknown"
    elif observed_reference_call_count<=5:
        difficulty="0-5"
    elif observed_reference_call_count<=11:
        difficulty="6-11"
    else:
        difficulty="12+"
    return {
      "task_sha256":sha256(task_id.encode()).hexdigest(),
      "scenario_sha256":sha256(task_id.split("_",1)[0].encode()).hexdigest(),
      "stage_version":PROTOCOL,
      "official_state_success":bool(official_state_success),
      "official_tests_pass":int(state_tests_pass),
      "official_tests_total":int(state_tests_total),
      "model_decision_steps":int(result.metrics.steps),
      "tool_calls":int(result.metrics.tool_calls),
      "tool_call_format_valid":valid_form,
      "tool_successes":int(result.metrics.tool_successes),
      "tool_errors":int(result.metrics.tool_errors),
      "model_invalid_tool_calls":int(result.metrics.invalid_tool_calls),
      "api_calls":api_calls,
      "api_http_failures":real_http_failures,
      "http_error_observations":len(http_error_indices),
      "http_error_followed_by_successful_api":recovered,
      "final_normal":bool(result.completed),
      "stop_reason":result.stop_reason,
      "reference_api_depth_bucket_for_postrun_only":difficulty,
      "raw_trace_or_task_text_exported":False
    }


def aggregate_dev57(records: list[dict], meta: dict, *,
                    expected_task_hashes: list[str]) -> dict:
    if (meta.get("protocol")!=PROTOCOL or
        meta.get("task_ids_sha256")!=OFFICIAL_DEV_IDS_SHA or
        meta.get("model_sha256")!=MODEL_GGUF_SHA or
        meta.get("n")!=N or
        meta.get("previously_exposed")!=EXPOSED_PILOTS or
        len(records)!=N or len(expected_task_hashes)!=N or
        len(set(expected_task_hashes))!=N):
        raise ValueError("full original frozen DEV57 evaluation required")
    expected_hash=sha256("\n".join(sorted(expected_task_hashes)).encode()).hexdigest()
    if expected_hash!=meta.get("task_hashes_sha256"):
        raise ValueError("frozen task hashes mismatch")
    actual=[r.get("task_sha256") for r in records]
    if len(set(actual))!=N or set(actual)!=set(expected_task_hashes):
        raise ValueError("missing or duplicated original DEV57 tasks")
    if any(r.get("stage_version")!=PROTOCOL or
           r.get("raw_trace_or_task_text_exported") is not False for r in records):
        raise ValueError("private source/trace or metric version drift")
    if len({r["scenario_sha256"] for r in records})!=SCENARIOS:
        raise ValueError("DEV task scenario identities changed")
    summ=lambda name: sum(int(x[name]) for x in records)
    calls=summ("tool_calls")
    api_calls=summ("api_calls")
    http=summ("api_http_failures")
    by_diff={}
    for label in (*DIFFICULTY_BUCKETS,"unknown"):
        subset=[x for x in records if x["reference_api_depth_bucket_for_postrun_only"]==label]
        if subset:
            by_diff[label]={"n":len(subset),
                           "state_successes":sum(x["official_state_success"] for x in subset),
                           "passed_state_tests":sum(x["official_tests_pass"] for x in subset),
                           "total_state_tests":sum(x["official_tests_total"] for x in subset)}
    return {
      "status":"COMPLETE_FIXED_APPWORLD_DEV57_STAGES",
      "protocol":PROTOCOL,"n":N,"scenarios":SCENARIOS,
      "previously_exposed_in_main":EXPOSED_PILOTS,
      "state_success":summ("official_state_success"),
      "task_success_rate":summ("official_state_success")/N,
      "official_state_test_pass":summ("official_tests_pass"),
      "official_state_test_total":summ("official_tests_total"),
      "model_decision_steps":summ("model_decision_steps"),
      "tool_calls":calls,
      "tool_format_validity_rate":summ("tool_call_format_valid")/calls if calls else None,
      "api_calls":api_calls,
      "api_http_failures":http,
      "api_http_failure_rate":http/api_calls if api_calls else None,
      "http_error_observations":summ("http_error_observations"),
      "recovered_http_errors":summ("http_error_followed_by_successful_api"),
      "post_http_recovery_rate":(
         summ("http_error_followed_by_successful_api")/
         summ("http_error_observations") if summ("http_error_observations") else None),
      "final_normal":summ("final_normal"),
      "final_rate":summ("final_normal")/N,
      "stop_reason_histogram":dict(Counter(x["stop_reason"] for x in records)),
      "difficulty_postrun_breakdown":by_diff,
      "no_training":True,"no_GAIA_score_claimed":True,
      "not_statistically_powered_for_5pp":True,
      "no_QA_K_inflation":True,
      "claim_boundary":"57 unique task outcomes clustered in 19 scenarios; 8 historical pilot tasks exposed; per-state-test counts nested within tasks are NOT independent observations. Do not claim a 5pt effect detected without paired confidence interval/power."
    }
