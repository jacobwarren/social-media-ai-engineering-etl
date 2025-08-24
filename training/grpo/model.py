from __future__ import annotations
from typing import Tuple

# Unsloth patch must happen before trainer/model construction elsewhere
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported  # type: ignore
PatchFastRL("GRPO", FastLanguageModel)


def load_base_model_and_tokenizer(
    model_name: str = "./sft-model",
    max_seq_length: int = 4096,
    lora_rank: int = 64,
    load_in_4bit: bool = True,
    fast_inference: bool = True,
    max_lora_rank: int | None = None,
    gpu_memory_utilization: float = 0.6,
) -> Tuple[object, object]:
    """
    Load the base model and tokenizer with Unsloth acceleration and prepare PEFT LoRA.
    Mirrors the logic currently used in 26-train-grpo.py.
    """
    if max_lora_rank is None:
        max_lora_rank = lora_rank

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=fast_inference,
        max_lora_rank=max_lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    return model, tokenizer

