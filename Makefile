UV := uv
DATASET_REPOSITORY := galilai-group/lewm-pusht
DATASET_DIRECTORY := data/raw
DATASET_NAME := pusht_expert_train.lance
LOCAL_CONFIG := configs/local.yaml
H200_CONFIG := configs/h200.yaml

.DEFAULT_GOAL := help
.PHONY: help sync download-data inspect-data validate-local validate-h200 format lint typecheck test check train-local evaluate-local train-lewm-local train-lewm-h200

help:
	@printf '%s\n' \
		'Available targets:' \
		'  sync           Install development, dataset, and upstream dependencies.' \
		'  download-data  Download or resume the official PushT Lance dataset.' \
		'  inspect-data   Stream one local PushT episode through the adapter.' \
		'  validate-local Validate the local smoke-test configuration.' \
		'  validate-h200  Validate the draft H200 configuration.' \
		'  train-local    Run a local smoke training with the local config.' \
		'  evaluate-local Run a local smoke evaluation with the local config.' \
		'  train-lewm-local Run a local LeWM baseline training with the two-term loss.' \
		'  train-lewm-h200 Run the full LeWM baseline on H200 (100 epochs, full data).' \
		'  format         Format the repository with Ruff.' \
		'  lint           Run Ruff lint checks.' \
		'  typecheck      Run strict mypy checks.' \
		'  test           Run the test suite.' \
		'  check          Format, lint, type-check, and test.'

sync:
	$(UV) sync --group dev --group data --extra upstream

download-data:
	$(UV) run --group data hf download $(DATASET_REPOSITORY) --repo-type dataset --local-dir $(DATASET_DIRECTORY) --include "$(DATASET_NAME)/**"

inspect-data:
	$(UV) run --extra upstream lewm-liquid-predictors inspect-pusht $(DATASET_DIRECTORY)/$(DATASET_NAME) --max-episodes 1

validate-local:
	$(UV) run lewm-liquid-predictors validate-config $(LOCAL_CONFIG)

validate-h200:
	$(UV) run lewm-liquid-predictors validate-config $(H200_CONFIG)

train-local:
	$(UV) run --extra upstream lewm-liquid-predictors train $(LOCAL_CONFIG) --max-episodes 8

evaluate-local:
	$(UV) run --extra upstream lewm-liquid-predictors evaluate $(LOCAL_CONFIG) --max-episodes 8

train-lewm-local:
	$(UV) run --extra upstream lewm-liquid-predictors train-lewm $(LOCAL_CONFIG) --max-episodes 4

train-lewm-h200:
	$(UV) run --extra upstream lewm-liquid-predictors train-lewm $(H200_CONFIG)

format:
	$(UV) run ruff format .

lint:
	$(UV) run ruff check .

typecheck:
	$(UV) run mypy src

test:
	$(UV) run pytest

check: format lint typecheck test
