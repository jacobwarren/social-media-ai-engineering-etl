from __future__ import annotations
from typing import List, Callable, Optional

from unsloth import is_bfloat16_supported  # type: ignore
from trl import GRPOConfig, GRPOTrainer  # type: ignore


def build_training_args(output_dir: str = "grpo-results") -> GRPOConfig:
    return GRPOConfig(
        # General settings
        use_vllm=True,
        learning_rate=1e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),

        # Batch settings
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,

        # Generation settings
        num_generations=8,
        max_prompt_length=4096,
        max_completion_length=1650,

        # Training schedule
        num_train_epochs=1,
        max_steps=250,
        save_steps=50,

        # Stability
        max_grad_norm=0.1,

        # Output
        report_to="wandb",
        output_dir=output_dir,
    )


def build_trainer(
    model,
    tokenizer,
    reward_funcs: List[Callable],
    training_args: GRPOConfig,
    train_dataset,
    eval_dataset=None,
) -> GRPOTrainer:
    return GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

