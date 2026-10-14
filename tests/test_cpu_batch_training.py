from gaia_small_agent.cli import build_parser
from gaia_small_agent.training.cpu_v3d_batches import (
    expected_completed_example_ids,
    partition_v3d_examples,
)
from gaia_small_agent.training.qlora import _training_runtime_defaults


def test_train_cli_exposes_resumable_step_batches():
    args = build_parser().parse_args([
        "train-qlora",
        "--data", "runs/oracle/train.jsonl",
        "--output", "runs/cpu-batch",
        "--max-steps", "2",
        "--save-steps", "1",
        "--gradient-accumulation-steps", "1",
        "--resume-from-checkpoint", "runs/previous/checkpoint-1",
        "--lora-target-modules", "q_proj,k_proj,v_proj,o_proj",
    ])

    assert args.max_steps == 2
    assert args.save_steps == 1
    assert args.gradient_accumulation_steps == 1
    assert args.resume_from_checkpoint == "runs/previous/checkpoint-1"
    assert args.lora_target_modules == "q_proj,k_proj,v_proj,o_proj"


def test_cpu_training_defaults_disable_gradient_checkpointing_and_use_bfloat16():
    defaults = _training_runtime_defaults(cuda_available=False, cuda_bf16_supported=False)

    assert defaults == {
        "compute_dtype_name": "bfloat16",
        "use_gradient_checkpointing": False,
        "trainer_loss_path": "native-cpu",
    }


def test_gpu_training_defaults_keep_gradient_checkpointing():
    defaults = _training_runtime_defaults(cuda_available=True, cuda_bf16_supported=True)

    assert defaults == {
        "compute_dtype_name": "bfloat16",
        "use_gradient_checkpointing": True,
        "trainer_loss_path": "trl-fused",
    }


def _fake_v3d_examples():
    return [
        {"metadata": {"example_id": f"example-{index:02d}"}}
        for index in range(24)
    ]


def test_v3d_cpu_partition_is_three_order_preserving_batches_of_eight():
    examples = _fake_v3d_examples()

    batches = partition_v3d_examples(examples)

    assert len(batches) == 3
    assert [len(batch) for batch in batches] == [8, 8, 8]
    assert [item["metadata"]["example_id"] for batch in batches for item in batch] == [
        item["metadata"]["example_id"] for item in examples
    ]


def test_v3d_cpu_resume_contract_requires_exact_completed_prefix():
    examples = _fake_v3d_examples()

    assert expected_completed_example_ids(examples, completed_batch_count=0) == []
    assert expected_completed_example_ids(examples, completed_batch_count=1) == [
        f"example-{index:02d}" for index in range(8)
    ]
    assert expected_completed_example_ids(examples, completed_batch_count=2) == [
        f"example-{index:02d}" for index in range(16)
    ]
