from gaia_small_agent.cli import build_parser


def test_gaia_eval_parser_accepts_shadow_partition():
    args = build_parser().parse_args([
        "gaia-eval",
        "--partition", "shadow",
        "--work-root", "runs/shadow",
    ])

    assert args.partition == "shadow"
    assert args.work_root == "runs/shadow"
