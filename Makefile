UV := uv
DATASET_REPOSITORY := galilai-group/lewm-pusht
DATASET_DIRECTORY := data/raw
DATASET_NAME := pusht_expert_train.lance
LOCAL_CONFIG := configs/local.yaml
H200_CONFIG := configs/h200.yaml
H200_PILOT_CONFIG := configs/h200-pilot.yaml
H200_SCREEN_CONFIG := configs/h200-screen.yaml
LEWM_OFFICIAL_CONFIG := configs/h200-lewm-official.yaml
H200_DECODER_CONFIG := configs/h200-decoder.yaml
LEWM_MODEL_REPOSITORY := quentinll/lewm-pusht
LEWM_MODEL_REVISION := 22b330c28c27ead4bfd1888615af1340e3fe9052
LEWM_CHECKPOINT_DIRECTORY := checkpoints/lewm-pusht
LEWM_CONFIG_SHA256 := 2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09
LEWM_WEIGHTS_SHA256 := 48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607

.DEFAULT_GOAL := help
.PHONY: help sync download-data download-lewm-checkpoint inspect-data validate-local validate-h200 validate-h200-pilot validate-h200-screen format lint typecheck test check train-local evaluate-local train-lewm-local train-lewm-h200 train-lewm-h200-pilot train-h200-screen evaluate-lewm-official train-h200-decoder render-h200-decoder-galleries clean

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
		'  train-h200-decoder Train only the post-hoc projected-CLS image decoder.' \
		'  render-h200-decoder-galleries Decode all saved predictor checkpoints without retraining.' \
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
	$(UV) run python -m lewm_liquid_predictors.artifacts https://huggingface.co/$(LEWM_MODEL_REPOSITORY)/resolve/$(LEWM_MODEL_REVISION)/config.json $(LEWM_CHECKPOINT_DIRECTORY)/config.json $(LEWM_CONFIG_SHA256)
	$(UV) run python -m lewm_liquid_predictors.artifacts https://huggingface.co/$(LEWM_MODEL_REPOSITORY)/resolve/$(LEWM_MODEL_REVISION)/weights.pt $(LEWM_CHECKPOINT_DIRECTORY)/weights.pt $(LEWM_WEIGHTS_SHA256)

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

train-h200-decoder:
	STABLEWM_HOME=$(CURDIR)/data/raw HF_HUB_OFFLINE=1 $(UV) run --extra upstream lewm-liquid-predictors train-decoder $(H200_DECODER_CONFIG) --checkpoint $(LEWM_CHECKPOINT_DIRECTORY)/weights.pt

render-h200-decoder-galleries:
	STABLEWM_HOME=$(CURDIR)/data/raw HF_HUB_OFFLINE=1 $(UV) run --extra upstream lewm-liquid-predictors render-decoder-galleries $(H200_DECODER_CONFIG) --checkpoint $(LEWM_CHECKPOINT_DIRECTORY)/weights.pt --predictor-root runs/h200-screen

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
