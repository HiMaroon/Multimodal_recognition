import json
import os
from pathlib import Path

RELEASE_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(os.environ.get("MMCELL_OUTPUT_ROOT", RELEASE_ROOT / "outputs"))
MANIFEST_PATH = OUTPUT_ROOT / "manifests" / "cells.csv"
FOLD_ASSIGNMENT_PATH = OUTPUT_ROOT / "manifests" / "cv.csv"
SUMMARY_DIR = OUTPUT_ROOT / "summaries"
PROTOCOL = json.loads((RELEASE_ROOT / "configs" / "protocol.json").read_text())
SELECTED_BACKBONES_PATH = OUTPUT_ROOT / "selected_backbones.json"
LOCKED_BACKBONES = PROTOCOL["selected_backbones"]
CLASS_ORDER = ["MCF7", "10A", "HCT"]
CLASS_TO_INDEX = {name: i for (i, name) in enumerate(CLASS_ORDER)}
MODALITIES = {"BFT": {}, "BFR": {}, "DFR": {}}
BACKBONES = {name: {"encoder_name": name} for name in PROTOCOL["backbones"]}
BACKBONES["swin_tiny"]["encoder_name"] = "swin_tiny_patch4_window7_224"
N_FOLDS = 3
SEED = PROTOCOL["cnn_seed"]
PATCH_SIZE = (224, 224)
EPOCHS = 30
EARLY_STOP_PATIENCE = 5
MIN_EPOCH_FOR_BEST = 5
TRAIN_AUG_REPEAT = 1
FIXED_BATCH_SIZE = 16
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))
WINDOWS_TRAIN_WORKERS = 2
WINDOWS_EVAL_WORKERS = 1
LR = 0.0001
WEIGHT_DECAY = 0.0001
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.9
SCHEDULER_MIN_LR = 1e-06
FUSION_HIDDEN_DIM = 1024
USE_AMP = True
USE_PIN_MEMORY = True
PRINT_EVERY_N_BATCHES = 25
SELECTION_PRIMARY_METRIC = "macro_f1"
