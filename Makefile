UV := uv
DATASET_REPOSITORY := galilai-group/lewm-pusht
DATASET_DIRECTORY := data/raw
DATASET_NAME := pusht_expert_train.lance
LOCAL_CONFIG := configs/local.yaml
H200_CONFIG := configs/h200.yaml

.DEFAULT_GOAL := help
.PHONY: help sync download-data inspect-data validate-local validate-h200 format lint typecheck test check

help:
	@printf '%s\n' \
		'Available targets:' \
		'  sync           Install development, dataset, and upstream dependencies.' \
		'  download-data  Download or resume the official PushT Lance dataset.' \
		'  inspect-data   Stream one local PushT episode through the adapter.' \
		'  validate-local Validate the local smoke-test configuration.' \
		'  validate-h200  Validate the draft H200 configuration.' \
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

format:
	$(UV) run ruff format .

lint:
	$(UV) run ruff check .

typecheck:
	$(UV) run mypy src

test:
	$(UV) run pytest

check: format lint typecheck test
