
# app/core/al_defaults.py

import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    _cfg = yaml.safe_load(f)


# Default values for active learning  inference
DEFAULT_INFERENCE_THRESHOLD = _cfg["DEFAULT_INFERENCE_THRESHOLD"]
DEFAULT_DENSITY_K = _cfg["DEFAULT_DENSITY_K"]
DEFAULT_COMPOSITE_WU = _cfg["DEFAULT_COMPOSITE_WU"]
DEFAULT_COMPOSITE_WD = _cfg["DEFAULT_COMPOSITE_WD"]
DEFAULT_COMPOSITE_WR = _cfg["DEFAULT_COMPOSITE_WR"]
DIVERSITY_HNSW_MIN_NL = _cfg["DIVERSITY_HNSW_MIN_NL"]
DIVERSITY_POOL_SIZE = _cfg["DIVERSITY_POOL_SIZE"]
RETRAIN_AFTER = _cfg["RETRAIN_AFTER"]

#Default values for training checkpoints
DEFAULT_EPOCHS = _cfg["DEFAULT_EPOCHS"]
DEFAULT_LEARNING_RATE = _cfg["DEFAULT_LEARNING_RATE"]
DEFAULT_BATCH_SIZE = _cfg["DEFAULT_BATCH_SIZE"]
DEFAULT_HIDDEN_DIM = _cfg["DEFAULT_HIDDEN_DIM"]
DEFAULT_DROPOUT = _cfg["DEFAULT_DROPOUT"]