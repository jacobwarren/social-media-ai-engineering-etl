PYTHON?=python
RUN_ID?=$(shell date +%Y%m%d_%H%M%S)
# Directory where this Makefile lives (absolute, trailing slash)
MAKEFILE_DIR:=$(dir $(abspath $(lastword $(MAKEFILE_LIST))))
# Default dirs/files relative to Makefile location so commands work from any CWD
BASE_DIR?=data/processed
DATASET?=example-dataset.jsonl
# Normalize user-provided or default paths against MAKEFILE_DIR if relative
# Ensure Python can import the local 'pipe' package no matter the CWD
# Prepend MAKEFILE_DIR to PYTHONPATH while preserving existing value
export PYTHONPATH := $(MAKEFILE_DIR)$(if $(PYTHONPATH),:$(PYTHONPATH),)

ifeq ($(filter /%,$(DATASET)),)
  DATASET_PATH:=$(MAKEFILE_DIR)$(DATASET)
else
  DATASET_PATH:=$(DATASET)
endif
ifeq ($(filter /%,$(BASE_DIR)),)
  BASE_DIR_PATH:=$(MAKEFILE_DIR)$(BASE_DIR)
else
  BASE_DIR_PATH:=$(BASE_DIR)
endif


.PHONY: setup setup-nlp run-tail run-core run-features run-e2e run-ablation lint fmt test smoke

setup:
	$(PYTHON) -m pip install -e .[dev]

setup-nlp:
	$(PYTHON) scripts/setup_nlp.py

# Tail-only: assumes earlier feature stages exist for $(RUN_ID)
run-tail:
	$(PYTHON) "$(MAKEFILE_DIR)17-writing-style.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)" --input "$(BASE_DIR_PATH)/$(RUN_ID)/15-clean-context.jsonl"
	$(PYTHON) "$(MAKEFILE_DIR)18-generate-prompts.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)"
	$(PYTHON) "$(MAKEFILE_DIR)22-generate-dataset.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)"
	$(PYTHON) "$(MAKEFILE_DIR)23-split.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)"

# Core ingestion + structure
run-core:
	$(PYTHON) "$(MAKEFILE_DIR)1-find-gradient.py" --input "$(DATASET_PATH)" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)" --report
	$(PYTHON) "$(MAKEFILE_DIR)2-label.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)"
	$(PYTHON) "$(MAKEFILE_DIR)3-extract-structures.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)"

# Feature stages needed by 18 (with cleaners for quality)
run-features:
	$(PYTHON) "$(MAKEFILE_DIR)6-extract-topics.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)"
	$(PYTHON) "$(MAKEFILE_DIR)7-clean-topics.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)"
	$(PYTHON) "$(MAKEFILE_DIR)9-extract-tone.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)"
	$(PYTHON) "$(MAKEFILE_DIR)11-extract-opinion.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)"
	$(PYTHON) "$(MAKEFILE_DIR)12-clean-opinions.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)"
	$(PYTHON) "$(MAKEFILE_DIR)14-extract-context.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)"
	$(PYTHON) "$(MAKEFILE_DIR)15-clean-context.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)"
	$(PYTHON) "$(MAKEFILE_DIR)17-writing-style.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)"

# Optional: run ablation (research)
run-ablation:
	$(PYTHON) 4-structure-micro-ablation.py --run-id $(RUN_ID)

# End-to-end from raw JSONL to splits
run-e2e: run-core run-features
	$(PYTHON) "$(MAKEFILE_DIR)18-generate-prompts.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)"
	$(PYTHON) "$(MAKEFILE_DIR)22-generate-dataset.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)"
	$(PYTHON) "$(MAKEFILE_DIR)23-split.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)" --disable-augmentation
	$(PYTHON) "$(MAKEFILE_DIR)25-train-sft.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)"
	$(PYTHON) "$(MAKEFILE_DIR)26-train-grpo.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)"
	$(PYTHON) "$(MAKEFILE_DIR)27-experiment.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)"

# Strict end-to-end: run all numbered stages in sequence, including 4 and 5
.PHONY: run-e2e-strict
run-e2e-strict:
	@echo "[STRICT] Running 01 → 27 sequentially (includes 04-ablation and 05-balance)"
	$(PYTHON) "$(MAKEFILE_DIR)1-find-gradient.py" --input "$(DATASET_PATH)" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)" --report
	$(PYTHON) "$(MAKEFILE_DIR)2-label.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)"
	$(PYTHON) "$(MAKEFILE_DIR)3-extract-structures.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)"
	# Optional research stage; included in strict for completeness
	$(PYTHON) "$(MAKEFILE_DIR)4-structure-micro-ablation.py" --run-id $(RUN_ID)
	# Legacy balance; still supported explicitly in strict mode
	$(PYTHON) "$(MAKEFILE_DIR)5-balance.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)"
	# Feature backbone
	$(PYTHON) "$(MAKEFILE_DIR)6-extract-topics.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)"
	$(PYTHON) "$(MAKEFILE_DIR)7-clean-topics.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)"
	$(PYTHON) "$(MAKEFILE_DIR)9-extract-tone.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)"
	$(PYTHON) "$(MAKEFILE_DIR)11-extract-opinion.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)"
	$(PYTHON) "$(MAKEFILE_DIR)12-clean-opinions.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)"
	$(PYTHON) "$(MAKEFILE_DIR)14-extract-context.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)"
	$(PYTHON) "$(MAKEFILE_DIR)15-clean-context.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)"
	$(PYTHON) "$(MAKEFILE_DIR)17-writing-style.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)"
	# Prompts and datasets
	$(PYTHON) "$(MAKEFILE_DIR)18-generate-prompts.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)"
	$(PYTHON) "$(MAKEFILE_DIR)22-generate-dataset.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)"
	$(PYTHON) "$(MAKEFILE_DIR)23-split.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)" --disable-augmentation
	# Training and experiments
	$(PYTHON) "$(MAKEFILE_DIR)25-train-sft.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)"
	$(PYTHON) "$(MAKEFILE_DIR)26-train-grpo.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)"
	$(PYTHON) "$(MAKEFILE_DIR)27-experiment.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR_PATH)"


lint:
	ruff check "$(MAKEFILE_DIR)"

fmt:
	black "$(MAKEFILE_DIR)" && isort "$(MAKEFILE_DIR)"

test:
	pytest -q "$(MAKEFILE_DIR)"

smoke:
	bash "$(MAKEFILE_DIR)tests/smoke_etl.sh" "$(MAKEFILE_DIR)"

# Evaluate modular rewards (CPU only)
EVAL_RUN?=$(shell date +%Y%m%d_%H%M%S)
.PHONY: eval-rewards

eval-rewards:
	$(PYTHON) "$(MAKEFILE_DIR)scripts/evaluate_rewards.py" --run-id $(EVAL_RUN) --base-dir "$(MAKEFILE_DIR)reports" --weights "$(MAKEFILE_DIR)training/rewards/weights.example.json"
# Prefect orchestration
.PHONY: flow
flow:
	$(PYTHON) "$(MAKEFILE_DIR)orchestration/prefect_flow.py" --run-id $${RUN_ID:-demo} --base-dir "$(BASE_DIR_PATH)" --reports-dir "$(MAKEFILE_DIR)reports" --tmp-dir "$(MAKEFILE_DIR)tmp" --weights "$(MAKEFILE_DIR)training/rewards/weights.example.json"

# Streamlit demo
.PHONY: demo

demo:
	streamlit run "$(MAKEFILE_DIR)app/score_app.py" -- --weights "$(MAKEFILE_DIR)training/rewards/weights.example.json"

# Training shortcuts
.PHONY: train-sft train-grpo experiment

train-sft:
	$(PYTHON) "$(MAKEFILE_DIR)25-train-sft.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR)" --models-dir "$(MAKEFILE_DIR)models"

train-grpo:
	cd "$(MAKEFILE_DIR)models/$(RUN_ID)" && $(PYTHON) "$(MAKEFILE_DIR)26-train-grpo.py" --run-id $(RUN_ID) --use-aggregator --weights "$(MAKEFILE_DIR)training/rewards/weights.example.json"

experiment:
	$(PYTHON) "$(MAKEFILE_DIR)27-experiment.py" --run-id $(RUN_ID) --base-dir "$(BASE_DIR)" --stage auto




