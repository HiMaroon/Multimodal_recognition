import argparse
import itertools
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
import cv_config as cfg
from cv_data import read_manifest, BatchSpec, ManifestDataset
from cv_models import FlexibleMultimodalModel
from expanded_models import build_single_model
from workflow import dump, selected
from protocol_guard import freeze, model_files, sha256, MODALITY_COLUMNS


def metrics(labels, probability):
    probability = np.asarray(probability, dtype=float)
    probability = probability / probability.sum(axis=1, keepdims=True)
    pred = probability.argmax(1)
    return {
        "macro_f1": f1_score(labels, pred, average="macro"),
        "macro_precision": precision_score(
            labels, pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(labels, pred, average="macro", zero_division=0),
        "macro_auc_ovr": roc_auc_score(
            labels, probability, multi_class="ovr", average="macro"
        ),
        "n_errors": int(np.sum(pred != labels)),
        "confusion_matrix": confusion_matrix(labels, pred, labels=[0, 1, 2]).tolist(),
    }


def load_model(root, device):
    qa = json.loads((root / "checkpoint_reload_qa.json").read_text())
    if qa["status"] != "pass":
        raise RuntimeError(f"Checkpoint QA failed: {root}")
    if qa.get("checkpoint_sha256") != sha256(root / "best_model_state.pt"):
        raise RuntimeError(f"Checkpoint changed after QA: {root}")
    architecture = json.loads((root / "architecture.json").read_text())
    if architecture.get("modality_columns") != MODALITY_COLUMNS:
        raise ValueError(f"Unverified physical modality mapping: {root}")
    mods = tuple(architecture["modalities"])
    backbones = architecture["backbones"]
    if len(mods) == 1:
        model = build_single_model(mods[0], backbones[mods[0]], pretrain=None)
    else:
        model = FlexibleMultimodalModel(
            mods,
            {m: cfg.BACKBONES[b]["encoder_name"] for (m, b) in backbones.items()},
            pretrain=None,
        )
    model.load_state_dict(
        torch.load(root / "best_model_state.pt", map_location="cpu", weights_only=True),
        strict=True,
    )
    return (model.to(device).eval(), mods)


def predict(model, array, device, batch_size=40):
    output = []
    with torch.inference_mode():
        for i in range(0, len(array), batch_size):
            x = torch.as_tensor(
                array[i : i + batch_size], dtype=torch.float32, device=device
            )
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                logits = model(x)["cls_pred"]
            output.append(torch.softmax(logits.float(), 1).cpu().numpy())
    return np.concatenate(output)


def evaluate(root, rows, device):
    (model, mods) = load_model(root, device)
    loader = DataLoader(
        ManifestDataset(rows, BatchSpec(mods, False)),
        batch_size=16,
        num_workers=cfg.NUM_WORKERS,
        shuffle=False,
    )
    probabilities = []
    for batch in loader:
        probabilities.append(predict(model, batch["images"].numpy(), device))
    del model
    return np.concatenate(probabilities)


def frame(rows, probability):
    result = pd.DataFrame(
        [
            {k: r[k] for k in ["sample_id", "parent_field_id", "cell_type", "label"]}
            for r in rows
        ]
    )
    result["label"] = result.label.astype(int)
    result["prediction"] = probability.argmax(1)
    for j in range(3):
        result[f"prob_{j}"] = probability[:, j]
    return result


def main(extra):
    p = argparse.ArgumentParser()
    p.add_argument("--all-backbones", action="store_true")
    args = p.parse_args(extra)
    chosen = selected()
    freeze_cnn(chosen)
    rows = [r for r in read_manifest() if r["original_split"] == "test"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = cfg.OUTPUT_ROOT / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    foldwise = []
    for size in [1, 2, 3]:
        for mods in itertools.combinations(list(cfg.MODALITIES), size):
            name = "+".join(mods)
            fold_prob = []
            for fold in range(3):
                root = cfg.OUTPUT_ROOT / ("cnn/single" if size == 1 else "cnn/fusion")
                root = root / mods[0] / chosen[mods[0]] if size == 1 else root / name
                probs = evaluate(root / f"fold_{fold}", rows, device)
                fold_prob.append(probs)
                frame(rows, probs).to_csv(out / f"{name}_fold{fold}.csv", index=False)
                foldwise.append(
                    {
                        "modalities": name,
                        "fold": fold,
                        **metrics(np.asarray([int(r["label"]) for r in rows]), probs),
                    }
                )
            probs = np.mean(fold_prob, axis=0)
            frame(rows, probs).to_csv(out / f"{name}.csv", index=False)
            summary.append(
                {
                    "modalities": name,
                    **metrics(np.asarray([int(r["label"]) for r in rows]), probs),
                }
            )
    dump(out / "ensemble_metrics.json", summary)
    dump(out / "fold_metrics.json", foldwise)
    if args.all_backbones:
        records = []
        for mod in cfg.MODALITIES:
            for backbone in cfg.BACKBONES:
                for fold in range(3):
                    root = (
                        cfg.OUTPUT_ROOT / "cnn/single" / mod / backbone / f"fold_{fold}"
                    )
                    probability = evaluate(root, rows, device)
                    result = json.loads((root / "experiment_result.json").read_text())
                    vals = result["val_overall"]
                    test = metrics(
                        np.asarray([int(r["label"]) for r in rows]), probability
                    )
                    records.append(
                        {
                            "modality": mod,
                            "backbone": backbone,
                            "fold": fold,
                            **{
                                f"val_{k}": vals[k]
                                for k in [
                                    "macro_f1",
                                    "macro_precision",
                                    "macro_recall",
                                    "macro_auc_ovr",
                                ]
                            },
                            **{
                                f"test_{k}": test[k]
                                for k in [
                                    "macro_f1",
                                    "macro_precision",
                                    "macro_recall",
                                    "macro_auc_ovr",
                                ]
                            },
                        }
                    )
        tab = pd.DataFrame(records)
        tab.to_csv(out / "backbone_fold_metrics.csv", index=False)
        tab.groupby(["modality", "backbone"]).agg(
            {c: ["mean", "std"] for c in tab if c.startswith(("val_", "test_"))}
        ).to_csv(out / "backbone_mean_sd.csv")


def freeze_cnn(chosen=None):
    chosen = selected() if chosen is None else chosen
    roots = [cfg.OUTPUT_ROOT / "cnn/single" / m / b / f"fold_{f}"
             for m in cfg.MODALITIES for b in cfg.BACKBONES for f in range(3)]
    roots += [cfg.OUTPUT_ROOT / "cnn/fusion" / "+".join(c) / f"fold_{f}"
              for size in (2, 3) for c in itertools.combinations(cfg.MODALITIES, size)
              for f in range(3)]
    for root in roots:
        qa = json.loads((root / "checkpoint_reload_qa.json").read_text())
        if qa.get("status") != "pass" or qa.get("checkpoint_sha256") != sha256(root / "best_model_state.pt"):
            raise RuntimeError(f"Unverified checkpoint: {root}")
        arch = json.loads((root / "architecture.json").read_text())
        if arch.get("modality_columns") != MODALITY_COLUMNS:
            raise ValueError(f"Modality identity mismatch: {root}")
        if len(arch["modalities"]) > 1 and any(arch["backbones"][m] != chosen[m] for m in arch["modalities"]):
            raise ValueError(f"Fusion architecture differs from validation selection: {root}")
    freeze("cnn", model_files(roots) + [cfg.SELECTED_BACKBONES_PATH, cfg.MANIFEST_PATH, cfg.FOLD_ASSIGNMENT_PATH])
