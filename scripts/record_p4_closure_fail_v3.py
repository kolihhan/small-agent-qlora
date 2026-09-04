"""Record the bounded v3 closure failure without retrying materialization."""
from __future__ import annotations
import hashlib, json, platform, subprocess, sys, time, urllib.request
from pathlib import Path
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT=Path(r"C:\Personal_Projects\ai-projects")
REPO=ROOT/"small-agent-qlora"; ART=ROOT/".runtimes"
RUNTIME=ART/"small-agent-qlora-p4-model-py311-v3"; WHEELHOUSE=ART/"small-agent-qlora-p4-model-wheelhouse-v3"; PROV=ART/"small-agent-qlora-p4-model-provenance-v3"
ROOTS={"transformers":"5.16.1","huggingface-hub":"1.28.0","tokenizers":"0.23.1","safetensors":"0.8.0","peft":"0.20.0","accelerate":"1.14.0","bitsandbytes":"0.50.2","datasets":"4.8.5"}
ENV=default_environment(); ENV.update({"implementation_name":"cpython","implementation_version":"3.11.9","os_name":"nt","platform_machine":"AMD64","platform_system":"Windows","python_full_version":"3.11.9","python_version":"3.11","sys_platform":"win32"})
def get(url):
    q=urllib.request.Request(url,headers={"User-Agent":"small-agent-qlora-p4/3"})
    with urllib.request.urlopen(q,timeout=60) as r:return json.load(r)
def rows(name,ver):
    out=[]
    for raw in get(f"https://pypi.org/pypi/{canonicalize_name(name)}/{ver}/json")["info"].get("requires_dist") or []:
        q=Requirement(raw); active=q.marker is None or q.marker.evaluate(ENV)
        out.append({"from":canonicalize_name(name),"from_version":ver,"raw":raw,"name":canonicalize_name(q.name),"specifier":str(q.specifier),"marker":str(q.marker) if q.marker else None,"active":bool(active),"extras":sorted(q.extras)})
    return out
def main():
    t=time.time(); assert PROV.exists() and not RUNTIME.exists() and WHEELHOUSE.exists() and not any(WHEELHOUSE.iterdir())
    edge=[]; constraints={}; conflicts=[]
    exact={canonicalize_name(k):Version(v) for k,v in ROOTS.items()}
    for n,v in ROOTS.items():
        for e in rows(n,v):
            if not e["active"]:continue
            constraints.setdefault(e["name"],[]).append(e["specifier"])
            if e["name"] in exact:
                ok=SpecifierSet(e["specifier"]).contains(exact[e["name"]],prereleases=True)
                e["root_exact_compatible"]=ok
                if not ok: conflicts.append(e)
            edge.append(e)
    torch_ver=subprocess.check_output([sys.executable,"-c","import torch;print(torch.__version__)"],text=True).strip()
    torch_req=subprocess.check_output([sys.executable,"-c","import importlib.metadata as m;print('\\n'.join(m.metadata('torch').get_all('Requires-Dist') or []))"],text=True).splitlines()
    torch_active=[]
    for raw in torch_req:
        q=Requirement(raw)
        if q.marker is None or q.marker.evaluate(ENV):
            active={"from":f"torch=={torch_ver}","raw":raw,"name":canonicalize_name(q.name),"specifier":str(q.specifier),"marker":str(q.marker) if q.marker else None}
            if active["name"]=="torch":continue
            torch_active.append(active); constraints.setdefault(active["name"],[]).append(active["specifier"])
    torch_edges=[]
    for e in edge:
        if e["name"]=="torch":
            ok=SpecifierSet(e["specifier"]).contains(Version(torch_ver),prereleases=True)
            torch_edges.append({"from":e["from"]+"=="+e["from_version"],"specifier":e["specifier"],"installed_version":torch_ver,"contains_prereleases_true":ok,"classification":"PASS" if ok else "GENUINE_TORCH_CONSTRAINT_FAIL"})
    reason="root exact conflict: transformers==5.16.1 requires huggingface-hub>=1.5.0,<2.0, but fixed root huggingface-hub==1.28.0 does not satisfy >=1.5.0"
    failure={"classification":"NON_TORCH_CLOSURE_FAIL","reason":reason,"conflicts":conflicts,"materialization":"ABORTED_BEFORE_VENV"}
    nodes={canonicalize_name(n):{"name":canonicalize_name(n),"version":v,"root":True,"origin":"official PyPI JSON metadata"} for n,v in ROOTS.items()}
    graph={"schema":"p4-model-runtime-v3","environment":ENV,"roots":ROOTS,"torch":{"version":torch_ver,"requires_dist":torch_req,"active_non_torch_requirements":torch_active,"active_torch_edges":torch_edges},"nodes":nodes,"active_edges":edge,"accumulated_constraints":constraints,"failure":failure,"non_torch_closure_resolved":False}
    (PROV/"graph.json").write_text(json.dumps(graph,indent=2)+"\n",encoding="utf-8")
    (PROV/"requirements-hashes.txt").write_text("# No lock: closure failed before materialization; torch external and absent\n",encoding="utf-8")
    (PROV/"wheel-hashes.txt").write_text("# No wheels downloaded: closure failed before materialization\n",encoding="utf-8")
    (PROV/"commands.json").write_text(json.dumps({"materialization_commands":[],"reason":"stopped before venv/download/install per bounded task"},indent=2)+"\n",encoding="utf-8")
    (PROV/"origins.json").write_text(json.dumps({"metadata_api":"https://pypi.org/pypi/<project>/<version>/json","sole_official_index":"https://pypi.org/simple","roots":ROOTS,"wheels":[],"torch_origin":"global system site package (external)"},indent=2)+"\n",encoding="utf-8")
    (PROV/"scoped-validation.json").write_text(json.dumps({"status":"NOT_RUN_MATERIALIZATION_ABORTED","runtime_exists":False,"wheelhouse_exists":False,"torch_external_version":torch_ver,"local_reachable_nodes":[],"issues":[]},indent=2)+"\n",encoding="utf-8")
    (PROV/"import-validation.txt").write_text("NOT_RUN: closure failed before venv/import validation\n",encoding="utf-8")
    (PROV/"bitsandbytes-native.json").write_text(json.dumps({"status":"NOT_RUN_MATERIALIZATION_ABORTED","dlls":[]},indent=2)+"\n",encoding="utf-8")
    (PROV/"pip-freeze.txt").write_text("NOT_RUN: venv was not materialized\n",encoding="utf-8")
    (PROV/"pip-check.txt").write_text("NOT_RUN: venv was not materialized\n",encoding="utf-8")
    status=subprocess.run(["git","status","--short"],cwd=REPO,text=True,capture_output=True).stdout
    inv={"repo":str(REPO),"git_status":status,"runtime":str(RUNTIME),"wheelhouse":str(WHEELHOUSE),"provenance":str(PROV),"files":sorted(p.name for p in PROV.iterdir())}
    (PROV/"inventory.json").write_text(json.dumps(inv,indent=2)+"\n",encoding="utf-8")
    (PROV/"times.json").write_text(json.dumps({"recorded_at_epoch":time.time(),"elapsed_seconds":time.time()-t,"materialization_started":False},indent=2)+"\n",encoding="utf-8")
    (PROV/"verdict.txt").write_text("NON_TORCH_CLOSURE_FAIL\n"+reason+"\nmaterialization=ABORTED_BEFORE_VENV\nno_retry_or_broaden=true\n",encoding="utf-8")
    (PROV/"run-summary.json").write_text(json.dumps({"verdict":"NON_TORCH_CLOSURE_FAIL","reason":reason,"dist_count":0,"wheelhouse_bytes":0,"runtime_exists":False,"wheelhouse_exists":False,"elapsed_seconds":time.time()-t},indent=2)+"\n",encoding="utf-8")
if __name__=="__main__":main()
