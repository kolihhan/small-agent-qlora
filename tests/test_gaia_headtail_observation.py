from gaia_small_agent.tools.base import ToolResult
from scripts.run_gaia_eval_shard import build_parser


def test_head_tail_preserves_budget_and_adds_tail_without_fabrication():
    original = "START " + "a" * 4500 + " END-EVIDENCE"
    result = ToolResult(True, original)
    normal, _ = result.bounded_observation(2000)
    balanced, _ = result.bounded_observation(2000, nonpaginated_head_tail=True)
    assert len(normal) <= 2000 and len(balanced) <= 2000
    assert normal.startswith("START ") and balanced.startswith("START ")
    assert not normal.endswith("END-EVIDENCE")
    assert balanced.endswith("END-EVIDENCE")
    assert "middle omitted" in balanced
    assert balanced != normal


def test_paginated_read_offset_never_skips_unseen_content():
    result = ToolResult(True, "z"*5000, metadata={
        "offset_chars": 500, "returned_chars": 5000,
        "total_chars": 9000, "next_offset_chars": 5500,
    })
    control, cm = result.bounded_observation(2000)
    treatment, tm = result.bounded_observation(2000, nonpaginated_head_tail=True)
    assert treatment == control
    assert tm == cm
    assert tm["next_offset_chars"] == 500 + tm["returned_chars"]
    assert 500 < tm["next_offset_chars"] < 5500


def test_tiny_and_exact_result_preserved():
    r = ToolResult(False, "abc", error_code="NOT_FOUND")
    assert r.bounded_observation(2000)[0] == r.bounded_observation(
        2000, nonpaginated_head_tail=True)[0]
    for max_chars in (1, 12, 30):
        value, _ = ToolResult(True, "x"*120).bounded_observation(
            max_chars, nonpaginated_head_tail=True)
        assert len(value) <= max_chars


def test_flag_is_default_off():
    parser = build_parser()
    argv = ["--hf-token","test","--shard-index","0","--work-root","x","--output","x/result.json"]
    assert not parser.parse_args(argv).nonpaginated_head_tail
    assert parser.parse_args(argv + ["--nonpaginated-head-tail"]).nonpaginated_head_tail
