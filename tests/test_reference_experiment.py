import copy
import json
from pathlib import Path

import pytest

from experiments.run_reference import validate_config


CONFIG_PATH = Path("experiments/reference_v0.1.json")


def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_reference_config_is_valid():
    validate_config(load_config())


def test_reference_config_rejects_unknown_optimizer():
    config = load_config()
    config["optimizers"] = list(config["optimizers"]) + ["UnknownOptimizer"]
    with pytest.raises(ValueError, match="unknown optimizers"):
        validate_config(config)


def test_reference_config_rejects_unknown_benchmark():
    config = load_config()
    config["benchmarks"] = list(config["benchmarks"]) + ["unknown"]
    with pytest.raises(ValueError, match="unknown benchmarks"):
        validate_config(config)


def test_reference_config_rejects_inconsistent_seed_policy():
    config = load_config()
    config["seeds"] = 4
    with pytest.raises(ValueError, match="seed_policy inconsistent"):
        validate_config(config)


def test_reference_config_rejects_nonpositive_dimension():
    config = load_config()
    config["dimensions"] = [10, 0]
    with pytest.raises(ValueError, match="dimensions"):
        validate_config(config)
