UV := uv
DATASET_REPOSITORY := galilai-group/lewm-pusht
DATASET_DIRECTORY := data/raw
DATASET_NAME := pusht_expert_train.lance
LOCAL_CONFIG := configs/local.yaml
H200_CONFIG := configs/h200.yaml
H200_PILOT_CONFIG := configs/h200-pilot.yaml
H200_SCREEN_CONFIG := configs/h200-screen.yaml
LEWM_OFFICIAL_CONFIG := configs/h200-lewm-official.yaml
LEWM_MODEL_REPOSITORY := quentinll/lewm-pusht
LEWM_MODEL_REVISION := 22b330c28c27ead4bfd1888615af1340e3fe9052
LEWM_CHECKPOINT_DIRECTORY := checkpoints/lewm-pusht

.DEFAULT_GOAL := help
.PHONY: help sync download-data download-lewm-checkpoint inspect-data validate-local validate-h200 validate-h200-pilot validate-h200-screen format lint typecheck test check train-local evaluate-local train-lewm-local train-lewm-h200 train-lewm-h200-pilot train-h200-screen evaluate-lewm-official clean

help:
	@printf '%s\n' \
		'Available targets:' \
		'  sync           Install development, dataset, and upstream dependencies.' \
		'  download-data  Download or resume the official PushT Lance dataset.' \
		'  download-lewm-checkpoint Download the pinned official PushT LeWM checkpoint.' \
		'  inspect-data   Stream one local PushT episode through the adapter.' \
		'  validate-local Validate the local smoke-test configuration.' \
		'  validate-h200  Validate the draft H200 configuration.' \
		'  validate-h200-pilot Validate the budgeted H200 pilot configuration.' \
		'  validate-h200-screen Validate the controlled H200 screening configuration.' \
		'  train-local    Run a local smoke training with the local config.' \
		'  evaluate-local Run a local smoke evaluation with the local config.' \
		'  train-lewm-local Run a local LeWM baseline training with the two-term loss.' \
		'  train-lewm-h200 Run the full LeWM baseline on H200 (100 epochs, full data).' \
		'  train-lewm-h200-pilot Run the 10%-data, 10-epoch H200 pilot (not a baseline result).' \
		'  train-h200-screen Run the three-seed, five-predictor H200 screening study.' \
		'  evaluate-lewm-official Evaluate the official full LeWM-JEPA checkpoint.' \
		'  format         Format the repository with Ruff.' \
		'  lint           Run Ruff lint checks.' \
		'  typecheck      Run strict mypy checks.' \
		'  test           Run the test suite.' \
		'  check          Format, lint, type-check, and test.' \
		'  clean          Remove runs/ and caches.'

sync:
	$(UV) sync --group dev --group data --extra upstream

download-data:
	$(UV) run --group data hf download $(DATASET_REPOSITORY) --repo-type dataset --local-dir $(DATASET_DIRECTORY) --include "$(DATASET_NAME)/**"

download-lewm-checkpoint:
	mkdir -p $(LEWM_CHECKPOINT_DIRECTORY)
	curl -L --fail --continue-at - --output $(LEWM_CHECKPOINT_DIRECTORY)/config.json https://huggingface.co/$(LEWM_MODEL_REPOSITORY)/resolve/$(LEWM_MODEL_REVISION)/config.json
	curl -L --fail --continue-at - --output $(LEWM_CHECKPOINT_DIRECTORY)/weights.pt https://huggingface.co/$(LEWM_MODEL_REPOSITORY)/resolve/$(LEWM_MODEL_REVISION)/weights.pt

inspect-data:
	$(UV) run --extra upstream lewm-liquid-predictors inspect-pusht $(DATASET_DIRECTORY)/$(DATASET_NAME) --max-episodes 1

validate-local:
	$(UV) run lewm-liquid-predictors validate-config $(LOCAL_CONFIG)

validate-h200:
	$(UV) run lewm-liquid-predictors validate-config $(H200_CONFIG)

validate-h200-pilot:
	$(UV) run lewm-liquid-predictors validate-config $(H200_PILOT_CONFIG)

validate-h200-screen:
	$(UV) run lewm-liquid-predictors validate-config $(H200_SCREEN_CONFIG)

train-local:
	$(UV) run --extra upstream lewm-liquid-predictors train $(LOCAL_CONFIG) --max-episodes 8

evaluate-local:
	$(UV) run --extra upstream lewm-liquid-predictors evaluate $(LOCAL_CONFIG) --max-episodes 8

train-lewm-local:
	STABLEWM_HOME=$(CURDIR)/data/raw HF_HUB_OFFLINE=1 $(UV) run --extra upstream lewm-liquid-predictors train-lewm $(LOCAL_CONFIG) --max-episodes 4

train-lewm-h200:
	STABLEWM_HOME=$(CURDIR)/data/raw HF_HUB_OFFLINE=1 $(UV) run --extra upstream lewm-liquid-predictors train-lewm $(H200_CONFIG)

train-lewm-h200-pilot:
	STABLEWM_HOME=$(CURDIR)/data/raw HF_HUB_OFFLINE=1 $(UV) run --extra upstream lewm-liquid-predictors train-lewm $(H200_PILOT_CONFIG)

train-h200-screen:
	STABLEWM_HOME=$(CURDIR)/data/raw HF_HUB_OFFLINE=1 $(UV) run --extra upstream lewm-liquid-predictors screen $(H200_SCREEN_CONFIG)

evaluate-lewm-official:
	STABLEWM_HOME=$(CURDIR)/data/raw HF_HUB_OFFLINE=1 $(UV) run --extra upstream lewm-liquid-predictors evaluate-lewm-official $(LEWM_OFFICIAL_CONFIG) --checkpoint $(LEWM_CHECKPOINT_DIRECTORY)/weights.pt

format:
	$(UV) run ruff format .

lint:
	$(UV) run ruff check .

typecheck:
	$(UV) run mypy src

test:
	$(UV) run pytest

check: format lint typecheck test

clean:
	rm -rf runs/* .mypy_cache .pytest_cache .ruff_cache
