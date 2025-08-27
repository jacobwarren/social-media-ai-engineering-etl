from __future__ import annotations
from typing import Tuple
import os

# Unsloth patch must happen before trainer/model construction elsewhere
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported  # type: ignore
PatchFastRL("GRPO", FastLanguageModel)

# Fallback to HF transformers for local paths
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
import torch


def _is_local_path(path: str) -> bool:
    try:
        return os.path.isdir(path) or path.startswith("./") or path.startswith("/")
    except Exception:
        return False


def load_base_model_and_tokenizer(
    model_name: str = "./sft-model",
    max_seq_length: int = 4096,
    lora_rank: int = 64,
    load_in_4bit: bool = True,
    fast_inference: bool = True,
    max_lora_rank: int | None = None,
    gpu_memory_utilization: float = 0.6,
) -> Tuple[object, object, bool]:
    """
    Load the base model and tokenizer.
    - Prefer Unsloth (vLLM-capable) even for local absolute paths.
    - If Unsloth rejects the local path, fall back to Transformers + PEFT.
    Returns (model, tokenizer, is_vllm_capable).
    """
    if max_lora_rank is None:
        max_lora_rank = lora_rank

    # Try Unsloth first for both Hub IDs and absolute local paths
    tried_unsloth = False
    if _is_local_path(model_name):
        tried_unsloth = True
        try:
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
            return model, tokenizer, True
        except Exception:
            # Fall through to Transformers path
            pass

    if not _is_local_path(model_name):
        # HF Hub model id: use Unsloth accelerated loader and Unsloth LoRA
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
        return model, tokenizer, True

    # Transformers fallback for local path
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
    )
    try:
        from peft import LoraConfig, get_peft_model  # type: ignore
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
    except Exception:
        pass
    return model, tokenizer, False

