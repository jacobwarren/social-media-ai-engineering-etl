PYTHON=python
RUN_ID?=$(shell date +%Y%m%d_%H%M%S)
BASE_DIR?=data/processed
DATASET?=../course-dataset-clean.jsonl


.PHONY: setup setup-nlp run-tail run-core run-features run-e2e run-ablation lint fmt test smoke

setup:
	$(PYTHON) -m pip install -e .[dev]

setup-nlp:
	$(PYTHON) scripts/setup_nlp.py

# Tail-only: assumes earlier feature stages exist for $(RUN_ID)
run-tail:
	$(PYTHON) 17-writing-style.py --run-id $(RUN_ID) --base-dir $(BASE_DIR) --input $(BASE_DIR)/$(RUN_ID)/15-clean-context.jsonl
	$(PYTHON) 18-generate-prompts.py --run-id $(RUN_ID) --base-dir $(BASE_DIR)
	$(PYTHON) 22-generate-dataset.py --run-id $(RUN_ID) --base-dir $(BASE_DIR)
	$(PYTHON) 23-split.py --run-id $(RUN_ID) --base-dir $(BASE_DIR)

# Core ingestion + structure
run-core:
	$(PYTHON) 1-find-gradient.py --input $(DATASET) --run-id $(RUN_ID) --base-dir $(BASE_DIR) --report
	$(PYTHON) 2-label.py --run-id $(RUN_ID) --base-dir $(BASE_DIR)
	$(PYTHON) 3-extract-structures.py --run-id $(RUN_ID) --base-dir $(BASE_DIR)

# Feature stages needed by 18 (with cleaners for quality)
run-features:
	$(PYTHON) 6-extract-topics.py --run-id $(RUN_ID) --base-dir $(BASE_DIR)
	$(PYTHON) 7-clean-topics.py --run-id $(RUN_ID) --base-dir $(BASE_DIR)
	$(PYTHON) 9-extract-tone.py --run-id $(RUN_ID) --base-dir $(BASE_DIR)
	$(PYTHON) 11-extract-opinion.py --run-id $(RUN_ID) --base-dir $(BASE_DIR)
	$(PYTHON) 12-clean-opinions.py --run-id $(RUN_ID) --base-dir $(BASE_DIR)
	$(PYTHON) 14-extract-context.py --run-id $(RUN_ID) --base-dir $(BASE_DIR)
	$(PYTHON) 15-clean-context.py --run-id $(RUN_ID) --base-dir $(BASE_DIR)
	$(PYTHON) 17-writing-style.py --run-id $(RUN_ID) --base-dir $(BASE_DIR)

# Optional: run ablation (research)
run-ablation:
	$(PYTHON) 4-structure-micro-ablation.py --run-id $(RUN_ID)

# End-to-end from raw JSONL to splits
run-e2e: run-core run-features
	$(PYTHON) 18-generate-prompts.py --run-id $(RUN_ID) --base-dir $(BASE_DIR)
	$(PYTHON) 22-generate-dataset.py --run-id $(RUN_ID) --base-dir $(BASE_DIR)
	$(PYTHON) 23-split.py --run-id $(RUN_ID) --base-dir $(BASE_DIR) --disable-augmentation

lint:
	ruff check .

fmt:
	black . && isort .

test:
	pytest -q

smoke:
	bash tests/smoke_etl.sh

# Evaluate modular rewards (CPU only)
EVAL_RUN?=$(shell date +%Y%m%d_%H%M%S)
.PHONY: eval-rewards

eval-rewards:
	$(PYTHON) scripts/evaluate_rewards.py --run-id $(EVAL_RUN) --base-dir reports --weights training/rewards/weights.example.json
# Prefect orchestration
.PHONY: flow
flow:
	$(PYTHON) orchestration/prefect_flow.py --run-id $${RUN_ID:-demo} --base-dir data/processed --reports-dir reports --tmp-dir tmp --weights training/rewards/weights.example.json

# Streamlit demo
.PHONY: demo

demo:
	streamlit run app/score_app.py -- --weights training/rewards/weights.example.json

# Training shortcuts
.PHONY: train-sft train-grpo experiment

train-sft:
	$(PYTHON) 25-train-sft.py --run-id $(RUN_ID) --base-dir $(BASE_DIR) --models-dir models

train-grpo:
	cd models/$(RUN_ID) && $(PYTHON) ../../pipe/26-train-grpo.py --run-id $(RUN_ID) --use-aggregator --weights ../../pipe/training/rewards/weights.example.json

experiment:
	$(PYTHON) 27-experiment.py --run-id $(RUN_ID) --base-dir $(BASE_DIR) --stage auto




