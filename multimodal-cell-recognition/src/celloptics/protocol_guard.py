"""Validate sample identity and freeze analysis inputs before test evaluation."""

import hashlib
import json
import importlib.metadata
import platform
from pathlib import Path
import cv_config as cfg

MODALITY_COLUMNS = {"BFR": "t_path", "BFT": "rb_path", "DFR": "rd_path"}
ACQUISITION_MODALITIES = {"T": "BFR", "RB": "BFT", "RD": "DFR"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def validate_fields(fields):
    identities, filenames = set(), set()
    for row in fields:
        fid = row["field_id"]
        if fid in identities:
            raise ValueError(f"Duplicate parent field: {fid}")
        identities.add(fid)
        if row["partition"] not in {"A", "B", "C"}:
            raise ValueError(f"Invalid partition: {fid}")
        if row["partition"] == "C" and row["cv_fold"] is not None:
            raise ValueError(f"Test field has a CV fold: {fid}")
        if row["partition"] != "C" and row["cv_fold"] not in {None, 0, 1, 2}:
            raise ValueError(f"Invalid CV fold: {fid}")
        if set(row["images"]) != set(MODALITY_COLUMNS):
            raise ValueError(f"Incomplete modality triplet: {fid}")
        for modality, name in row["images"].items():
            if name in filenames or f"-{modality}-" not in name:
                raise ValueError(f"Duplicate image or incorrect modality filename: {name}")
            filenames.add(name)


def validate_cells(rows, development_only=False):
    fields = json.loads((cfg.RELEASE_ROOT / "configs/fields.json").read_text())
    validate_fields(fields)
    lookup = {r["field_id"]: r for r in fields}
    samples, paths = set(), set()
    for row in rows:
        sid, fid = row["sample_id"], row["parent_field_id"]
        if sid in samples or fid not in lookup:
            raise ValueError(f"Duplicate sample or unknown parent field: {sid}")
        samples.add(sid)
        source = lookup[fid]
        split = {"A": "train", "B": "val", "C": "test"}[source["partition"]]
        role = "external_test" if split == "test" else "main"
        if row["original_split"] != split or row["dataset_role"] != role:
            raise ValueError(f"Partition mismatch: {sid}")
        if development_only and split == "test":
            raise ValueError(f"Test sample in development manifest: {sid}")
        if row["cell_type"] != source["cell_type"] or int(row["label"]) != cfg.CLASS_TO_INDEX[source["cell_type"]]:
            raise ValueError(f"Class identity mismatch: {sid}")
        value = row.get("fold")
        fold = None if value in (None, "", "None") else int(value)
        if fold != source["cv_fold"] or (split != "test" and fold is None):
            raise ValueError(f"Parent-field fold mismatch: {sid}")
        for column in MODALITY_COLUMNS.values():
            path = str(Path(row[column]).resolve())
            if path in paths:
                raise ValueError(f"Repeated image across samples or modalities: {path}")
            paths.add(path)
    return rows


def validate_training_rows(train, validation):
    validate_cells(train, development_only=True)
    validate_cells(validation, development_only=True)
    if not train or not validation:
        raise ValueError("Training and validation partitions must be nonempty")
    for rows in (train, validation):
        if {int(r["label"]) for r in rows} != {0, 1, 2}:
            raise ValueError("Training and validation must each contain all three classes")
    if {r["parent_field_id"] for r in train} & {r["parent_field_id"] for r in validation}:
        raise ValueError("Training and validation share a parent field")


def bind_run():
    files = [cfg.RELEASE_ROOT / "run.py", cfg.RELEASE_ROOT / "requirements.txt"]
    files += sorted((cfg.RELEASE_ROOT / "configs").glob("*.json"))
    files += sorted((cfg.RELEASE_ROOT / "src/celloptics").glob("*.py"))
    contract = {p.relative_to(cfg.RELEASE_ROOT).as_posix(): sha256(p) for p in files}
    contract["environment"] = {
        "python": platform.python_version(), "platform": platform.platform(),
        "workers": cfg.NUM_WORKERS,
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "torchvision", "numpy", "scikit-learn", "scipy", "cellpose", "timm", "segmentation-models-pytorch")},
    }
    path = cfg.OUTPUT_ROOT / "run_contract.json"
    if path.exists():
        if json.loads(path.read_text()) != contract:
            raise RuntimeError("Code or configuration changed. Use a new --output directory.")
    else:
        if cfg.OUTPUT_ROOT.exists() and any(cfg.OUTPUT_ROOT.iterdir()):
            raise RuntimeError("Unversioned outputs exist. Use a new --output directory.")
        save_json(path, contract)


def require_development(stage):
    if (cfg.OUTPUT_ROOT / "locks" / f"{stage}.json").exists():
        raise RuntimeError(f"{stage} is frozen for test evaluation. Development is closed for this run.")


def freeze(stage, files):
    contract = {str(Path(p).resolve()): sha256(p) for p in sorted(set(files))}
    path = cfg.OUTPUT_ROOT / "locks" / f"{stage}.json"
    if path.exists():
        if json.loads(path.read_text()) != contract:
            raise RuntimeError(f"Frozen {stage} inputs changed")
    else:
        save_json(path, contract)


def model_files(roots):
    names = ["architecture.json", "experiment_result.json", "checkpoint_reload_qa.json", "best_model_state.pt"]
    return [p / name for p in roots for name in names]
