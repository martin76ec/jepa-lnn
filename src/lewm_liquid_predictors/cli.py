"""Small executable entry points for configuration and data smoke tests."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from lewm_liquid_predictors.data import load_pusht_lance_episodes
from lewm_liquid_predictors.utils import load_config


def main(arguments: Sequence[str] | None = None) -> int:
    """Run a project command and return its process exit status."""
    parser = argparse.ArgumentParser(prog="lewm-liquid-predictors")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config")
    validate.add_argument("config", type=Path)

    inspect = subparsers.add_parser("inspect-pusht")
    inspect.add_argument("path")
    inspect.add_argument("--max-episodes", type=int, default=1)

    parsed = parser.parse_args(arguments)
    if parsed.command == "validate-config":
        config = load_config(parsed.config)
        print(json.dumps(asdict(config), default=str, indent=2, sort_keys=True))
        return 0
    trajectories = load_pusht_lance_episodes(parsed.path, max_episodes=parsed.max_episodes)
    print(
        json.dumps(
            [
                {
                    "episode_id": trajectory.episode_id,
                    "observations_shape": list(trajectory.observations.shape),
                    "actions_shape": list(trajectory.actions.shape),
                }
                for trajectory in trajectories
            ],
            indent=2,
        )
    )
    return 0
