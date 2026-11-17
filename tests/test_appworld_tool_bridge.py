"""No AppWorld installation required: public docs and tool budget contract."""
from pathlib import Path
from types import SimpleNamespace
from gaia_small_agent.tools.appworld_tool import AppWorldDocsTool, AppWorldCallTool, allowed_public_api

class FakeRequester:
    def __init__(self):self.calls=[]
    def request(self,**kwargs):
        self.calls.append(kwargs)
        return {"ok":True,"items":[1]}

def world():
    docs={"spotify":{"search_tracks":{"description":"Search tracks","parameters":[{"name":"query","type":"string","required":True}]},
                    "create_playlist":{"description":"Create playlist","parameters":[]}}}
    return SimpleNamespace(task=SimpleNamespace(allowed_apps=["spotify"],api_docs=docs),
                           requester=FakeRequester())

def test_no_access_to_unavailable_apps_or_private_oracle():
    w=world()
    assert allowed_public_api(w,"spotify","search_tracks")
    for app,api in (("admin","list_users"),("api_docs","get"),("gmail","send"),
                    ("spotify","__dict__"),("supervisor","get_answer")):
        assert not allowed_public_api(w,app,api)

def test_public_docs_tool_never_shows_reference_or_private_fields():
    w=world();t=AppWorldDocsTool(w)
    assert "spotify" in t.run({},Path(".")).content
    assert t.run({"app":"gmail"},Path(".")).error_code=="APP_DENIED"
    x=t.run({"app":"spotify","api":"search_tracks"},Path("."))
    assert x.ok and "query" in x.content
    assert "ground_truth" not in x.content and "evaluation_code" not in x.content

def test_tool_only_calls_allowlisted_business_api_and_enforces_cap():
    w=world();t=AppWorldCallTool(w,max_calls=1)
    good=t.run({"app":"spotify","api":"search_tracks","arguments":{"query":"a"}},Path("."))
    assert good.ok and len(w.requester.calls)==1
    assert w.requester.calls[0]["_api_name"]=="search_tracks"
    assert not t.run({"app":"spotify","api":"search_tracks","arguments":{"query":"a"}},Path(".")).ok
    assert len(w.requester.calls)==1
    assert t.run({"app":"admin","api":"list_users","arguments":{}},Path(".")).error_code=="API_DENIED"

def test_only_supervisor_completion_allowed():
    w=world()
    t=AppWorldCallTool(w)
    assert allowed_public_api(w,"supervisor","complete_task")
    assert t.run({"app":"supervisor","api":"complete_task","arguments":{}},Path(".")).ok
    assert not allowed_public_api(w,"supervisor","see_reference")


def test_opt_in_documented_supervisor_get_only_never_writes_or_hidden_oracle():
    w=world()
    w.task.api_docs["supervisor"]={
      "show_profile":{"method":"GET","path":"/supervisor/show_profile",
                      "description":"Public user profile"},
      "secret_answer":{"method":"POST","path":"/supervisor/secret_answer",
                       "description":"Private"},
    }
    assert not allowed_public_api(w,"supervisor","show_profile")
    assert allowed_public_api(w,"supervisor","show_profile",allow_supervisor_public_reads=True)
    assert not allowed_public_api(w,"supervisor","secret_answer",allow_supervisor_public_reads=True)
    docs=AppWorldDocsTool(w,allow_supervisor_public_reads=True)
    available=docs.run({"app":"supervisor"},Path(".")).content
    assert "show_profile" in available and "secret_answer" not in available
    tool=AppWorldCallTool(w,allow_supervisor_public_reads=True)
    assert tool.run({"app":"supervisor","api":"show_profile","arguments":{}},Path(".")).ok
    assert tool.run({"app":"supervisor","api":"secret_answer","arguments":{}},Path(".")).error_code=="API_DENIED"
