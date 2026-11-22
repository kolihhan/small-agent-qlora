from scripts.audit_gaia_capability_census import analyze, evidence_for


def row(correct, tag=None, code=None):
    trace = [
        {"kind": "tool_call", "data": {"name": "read", "arguments": {"source": "do-not-export"}}},
        {"kind": "tool_result", "data": {"name": "read", "ok": code is None,
                                         "error_code": code, "content": "DO-NOT-EXPORT"}},
    ]
    return {"correct": correct, "completed": True, "capability_gap": tag, "trace": trace}


def test_census_counts_modalities_and_does_not_expose_trace():
    a = {"id%03d" % n: row(n < 20, "semantic_image_unsupported" if n < 10 else None,
                             "UNSUPPORTED_BINARY" if n < 10 else None)
         for n in range(100)}
    b = {k: dict(v) for k,v in a.items()}
    b["id010"] = row(False)
    report = analyze(a, b)
    assert report["base"]["semantic_image_unsupported"]["tasks"] == 10
    assert report["base"]["semantic_image_unsupported"]["wrong"] == 0
    assert report["base"]["no_tagged_gap"]["wrong"] == 80
    assert report["qlora_v8"]["no_tagged_gap"]["wrong"] == 81
    assert "do-not-export" not in str(report)
    assert report["outcome_by_attachment_category"]["no_tagged_gap"]["base_only"] == 1


def test_binary_read_error_is_a_technical_signal_not_proof_of_necessity():
    evidence = evidence_for(row(False, "semantic_audio_unsupported", "UNSUPPORTED_BINARY"))
    assert evidence["read_attempted"]
    assert evidence["known_read_errors"] == ["UNSUPPORTED_BINARY"]
