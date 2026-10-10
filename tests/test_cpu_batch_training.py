from gaia_small_agent.cli import build_parser
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
