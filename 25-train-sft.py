import os
try:
    import wandb
except Exception:
    wandb = None  # type: ignore
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from transformers import TrainingArguments
from trl import SFTTrainer

from utils.seed import set_global_seed
from utils.logging_setup import init_pipeline_logging
from utils.manifest import read_manifest, discover_input, should_skip, compute_hash, update_stage
from utils.run_id import get_last_run_id

# CLI for paths and run IDs (non-breaking defaults)
import argparse
parser = argparse.ArgumentParser(description="SFT training")
parser.add_argument("--csv", default="sft.csv")
parser.add_argument("--run-id", default=None)
parser.add_argument("--base-dir", default="data/processed")
parser.add_argument("--models-dir", default="models")
parser.add_argument("--seed", type=int, default=3407)
args = parser.parse_args()

logger = init_pipeline_logging("phase3.train_sft", None, "25-train-sft")

# Resolve latest run-id if requested
if args.run_id == "latest":
    latest = get_last_run_id(args.base_dir)
    if latest:
        args.run_id = latest

# Discover dataset CSV via manifest when run-id is provided
csv_path = args.csv
if args.run_id:
    try:
        # Resolve run-id and base dir (ensures directory exists); ignore returned path
        from utils.io import resolve_io
        _, _, _ = resolve_io(stage="25-train-sft", run_id=args.run_id, base_dir=args.base_dir, explicit_in=csv_path, prior_stage="23-split", std_name=None)
        # Prefer SFT CSV from 23-split outputs
        m = read_manifest(args.run_id, args.base_dir)
        stage = m.get("stages", {}).get("23-split", {})
        outs = stage.get("outputs") or stage.get("output") or []
        candidates = outs if isinstance(outs, list) else ([outs] if isinstance(outs, str) else [])
        preferred = [p for p in candidates if isinstance(p, str) and p.endswith("23-sft.csv")] or candidates
        if preferred:
            csv_path = preferred[0]
        logger.info(f"Using training CSV: {csv_path}")
    except Exception:
        logger.warning("Falling back to --csv; manifest discovery failed")

# Seed once
set_global_seed(args.seed)

##############################################################################
# 1) Setup Weights & Biases (optional)
##############################################################################
if wandb is not None:
    run = wandb.init(project='GrowLlama', job_type="training", anonymous="allow")
else:
    logger.warning("wandb not available; proceeding without logging to Weights & Biases")


##############################################################################
# 2) Load Base Model
##############################################################################
base_model = "Qwen/Qwen3-14B"
# final name for saving
if args.run_id:
    os.makedirs(os.path.join(args.models_dir, args.run_id), exist_ok=True)
    new_model = os.path.join(args.models_dir, args.run_id, "sft-model")
else:
    new_model = "sft-model"

# Early idempotent skip when model exists
if args.run_id and os.path.isdir(new_model):
    logger.info(f"Model already exists at {new_model}; skipping training")
    raise SystemExit(0)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model,
    load_in_4bit=True,         # or False if you want full precision
    dtype=None,                # or "float16", "bfloat16", etc.
    attn_implementation="flash_attention_2",
    max_seq_length=8192
)

##############################################################################
# 3) Load CSV Data
##############################################################################
# Use the discovered csv_path (do not overwrite with args.csv)
raw_dataset = load_dataset("csv", data_files=csv_path)["train"]

##############################################################################
# 4) Filter out empty/whitespace rows early and manifest idempotency
##############################################################################
# If run-id, compute signature and check should_skip
if args.run_id:
    try:
        sig = compute_hash([csv_path], {"stage": 25, "base_model": base_model})
        manifest = read_manifest(args.run_id, args.base_dir)
        if should_skip(manifest, "25-train-sft", sig, [new_model]):
            logger.info(f"Skipping 25-train-sft; up-to-date at {new_model}")
            raise SystemExit(0)
    except Exception:
        pass
def non_empty_prompt_chosen(example):
    """
    Returns False if either prompt or chosen is None/whitespace only.
    True otherwise, so the row is kept.
    """
    p = example.get("prompt", "")
    c = example.get("chosen", "")
    if not isinstance(p, str):
        p = str(p) if p else ""
    if not isinstance(c, str):
        c = str(c) if c else ""
    return (len(p.strip()) > 0) and (len(c.strip()) > 0)

filtered_dataset = raw_dataset.filter(non_empty_prompt_chosen)

##############################################################################
# 4a) Idempotent manifest update after success
##############################################################################
# Defer update until after save

print(
    "Original dataset size:",
    len(raw_dataset),
    "- After removing empty prompt/response:",
    len(filtered_dataset)
)

##############################################################################
# 5) Pick the “llama-3.1” conversation template
##############################################################################
tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen-2.5",
)

##############################################################################
# 6) Format prompts once
##############################################################################
def formatting_prompts_func(examples):
    """Convert each (prompt, chosen) row into a 2-turn conversation, then apply
       the Llama 3.1 chat template exactly once."""
    prompt = examples["prompt"]
    chosen = examples["chosen"]

    # Guarantee they are strings, but we've already filtered out empties
    prompt = str(prompt)
    chosen = str(chosen)

    # Create conversation roles (2 turns: user -> assistant)
    conversation = [
        {'role': 'user', 'content': prompt},
        {'role': 'assistant', 'content': chosen},
    ]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

dataset = filtered_dataset.map(formatting_prompts_func)

print("Template applied example:")
print(dataset[0]["text"][:200] + "...")

##############################################################################
# 7) Split train/test
##############################################################################
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

##############################################################################
# 8) Attach LoRA adapters
##############################################################################
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None
)

##############################################################################
# 9) Compute max seq length
##############################################################################
import numpy as np

def calculate_max_seq_length(percentile=100):
    lengths = []
    for txt in dataset["text"]:
        enc = tokenizer(txt, return_tensors="pt")
        lengths.append(enc["input_ids"].shape[1])
    max_length = int(np.percentile(lengths, percentile))
    max_model_length = getattr(tokenizer, "model_max_length", 8192)
    return min(max_length, max_model_length)

max_seq_length = calculate_max_seq_length()
print(f"Calculated max_seq_length: {max_seq_length}")

##############################################################################
# 10) Filter out examples that exceed max_seq_length
##############################################################################
def filter_by_max_length(ex):
    try:
        enc = tokenizer(ex["text"], return_tensors="pt")
        return enc["input_ids"].shape[1] <= max_seq_length
    except Exception as exc:
        print(f"Warning: Could not tokenize example, keeping by default. Error: {exc}")
        return True

train_dataset_f = train_dataset.filter(filter_by_max_length)
eval_dataset_f = eval_dataset.filter(filter_by_max_length)

##############################################################################
# 11) Setup TrainingArguments
##############################################################################
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./sft-results",
    num_train_epochs=1,
    do_eval=True,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=4,
    optim="adamw_torch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=2e-4,
    load_best_model_at_end=True,
    fp16=False,
    bf16=True,
    max_steps=-1,
    warmup_ratio=0.1,
    weight_decay=0.01,
    max_grad_norm=1,
    lr_scheduler_type="cosine",
    report_to="wandb",
    run_name="GrowLlama-run",
)

##############################################################################
# 12) Initialize SFTTrainer
##############################################################################
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset_f,
    eval_dataset=eval_dataset_f,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=False,
    args=training_args,
    neftune_noise_alpha=0,  # Turn off noising
    dataset_num_proc=1,
)

trainer.model.config.use_cache = False

##############################################################################
# 13) Only train on the assistant’s responses
##############################################################################
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

##############################################################################
# 14) Quick sanity-check: ensure some tokens are not -100
##############################################################################
labels = trainer.train_dataset[0]["labels"]
non_masked_count = sum(1 for x in labels if x != -100)
# Write a minimal model card
try:
    import json, time
    card = {
        "run_id": args.run_id,
        "base_model": base_model,
        "seed": 3407,
        "dataset_csv": csv_path,
        "max_seq_length": max_seq_length,
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    out_dir = os.path.dirname(new_model) if os.path.dirname(new_model) else "."
    with open(os.path.join(out_dir, "model_card.json"), "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
except Exception:
    pass

print(f"[Sanity Check] Non-masked tokens in first example: {non_masked_count}")
if non_masked_count > 0:
    sample_ids = [x for x in labels if x != -100][:50]
    print("Sample unmasked text:", tokenizer.decode(sample_ids))
else:
    print("WARNING: All tokens masked! Possibly incorrect instruction_part/response_part.")

##############################################################################
# 15) Train
##############################################################################
trainer_stats = trainer.train()

##############################################################################
# 16) Save Model
##############################################################################
model.save_pretrained_merged(new_model, tokenizer, save_method="merged_16bit")

# After saving, update manifest if run-id
try:
    if args.run_id:
        manifest = read_manifest(args.run_id, args.base_dir)
        sig = compute_hash([csv_path], {"stage": 25, "base_model": base_model})
        update_stage(
            args.run_id,
            args.base_dir,
            manifest,
            stage_name="25-train-sft",
            input_path=csv_path,
            outputs=[new_model],
            signature=sig,
            extra={"epochs": training_args.num_train_epochs, "dataset_rows": len(dataset)}
        )
except Exception:
    pass


if wandb is not None:
    wandb.finish()
