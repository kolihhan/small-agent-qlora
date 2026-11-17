"""CPU unit tests for TRAIN-only gold reference/public AppWorld tool contract."""
from types import SimpleNamespace
from scripts import audit_appworld_train_api_bridge as m


def _docs():
    return {
      "spotify":{
          "search_tracks":{"method":"GET","path":"/spotify/search_tracks",
                           "parameters":[{"name":"query"}]},
          "get_playlist":{"method":"GET","path":"/spotify/playlists/{playlist_id}",
                          "parameters":[{"name":"playlist_id"}]},
          "add_song":{"method":"POST","path":"/spotify/playlists/{playlist_id}/songs",
                      "parameters":[{"name":"track_id"}]}
      },
      "supervisor":{
          "complete_task":{"method":"POST","path":"/supervisor/complete_task"},
          "show_solution":{"method":"GET","path":"/supervisor/show_solution"}
      },
      "admin":{"dump":{"method":"GET","path":"/admin/dump"}}
    }


def test_public_route_mapping_and_private_supervisor_or_admin_deny():
    routes=m.public_route_lookup(_docs(),["spotify","admin"])
    types=[(app,api) for group in routes.values() for app,api,_,_ in group]
    assert ("spotify","search_tracks") in types
    assert ("supervisor","complete_task") in types
    assert ("supervisor","show_solution") not in types
    assert all(app!="admin" for app,api in types)


def test_original_method_url_body_and_path_parameters_recover():
    routes=m.public_route_lookup(_docs(),["spotify"])
    state,action=m.match_public_request({
        "method":"get","url":"http://spotify.com/spotify/playlists/playlist42",
        "data":{"skip":1}},routes)
    assert state=="mapped"
    assert action=={"app":"spotify","api":"get_playlist",
                    "arguments":{"skip":1,"playlist_id":"playlist42"}}
    s2,a2=m.match_public_request({"method":"post",
             "url":"/spotify/playlists/playlist42/songs",
             "data":{"track_id":"song42"}},routes)
    assert s2=="mapped" and a2["api"]=="add_song"
    assert a2["arguments"]["playlist_id"]=="playlist42"


def test_bad_reference_and_unknown_routes_abstain():
    routes=m.public_route_lookup(_docs(),["spotify"])
    assert m.match_public_request({"method":"get","url":"/admin/dump","data":{}},routes)[0]=="unmapped"
    assert m.match_public_request({"method":"get","url":"/spotify/search_tracks","data":"bad"},routes)[0]=="unmapped"
    assert m.match_public_request({},routes)[0]=="invalid_reference"


def test_reproduction_prior_source_only_selection():
    def calls(i):
        return [{"url":f"/spotify/read/{j}","method":"get","data":{}} for j in range(9)]+[
          {"url":f"/venmo/search/{j}","method":"get","data":{}} for j in range(8)]
    data=[(f"scene{i}_1",SimpleNamespace(api_calls=calls(i))) for i in range(20)]
    assert len(m.choose_train_entries(data))==8
    assert m.choose_train_entries(data)==m.choose_train_entries(list(reversed(data)))


def test_unmapped_actions_are_categorized_not_auto_skipped_when_business():
    assert m.request_category({"method":"post","url":"/admin/initialize","data":{}})=="admin"
    assert m.request_category({"method":"get","url":"https://x.org/api_docs/show","data":{}})=="api_docs"
    assert m.request_category({"method":"post","url":"/supervisor/complete_task","data":{}})=="supervisor"
    assert m.request_category({"method":"get","url":"/spotify/search","data":{}})=="business"
    assert m.request_category({"url":""})=="malformed"


def test_public_supervisor_get_is_opt_in_only_and_never_private_write():
    d=_docs()
    d["supervisor"]["show_profile"]={"method":"GET","path":"/supervisor/show_profile"}
    d["supervisor"]["delete_database"]={"method":"POST","path":"/supervisor/delete_database"}
    original=m.public_route_lookup(d,["spotify"])
    expanded=m.public_route_lookup(d,["spotify"],allow_supervisor_public_reads=True)
    expected={"method":"get","url":"/supervisor/show_profile","data":{}}
    assert m.match_public_request(expected,original)[0]=="unmapped"
    assert m.match_public_request(expected,expanded)[0]=="mapped"
    assert m.match_public_request({"method":"post","url":"/supervisor/delete_database","data":{}},expanded)[0]=="unmapped"
