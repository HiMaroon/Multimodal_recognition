import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import cv_config as cfg
from protocol_guard import bind_run, require_development, validate_fields, validate_cells, sha256, MODALITY_COLUMNS


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def table(path, rows):
    if not rows:
        raise ValueError(f"No records to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def data_args(extra):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data", type=Path, default=cfg.RELEASE_ROOT.parent / "whole_field_data"
    )
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args(extra)


def field_rows():
    rows = json.loads((cfg.RELEASE_ROOT / "configs/fields.json").read_text())
    validate_fields(rows)
    return rows


def verify_data(extra):
    from PIL import Image

    args = data_args(extra)
    records = field_rows()
    failures = []
    for field in records:
        shapes = []
        for modality, name in field["images"].items():
            path = args.data / name
            if not path.exists():
                failures.append(f"Missing {name}")
                continue
            with Image.open(path) as image:
                shapes.append(image.size)
                if image.mode != "RGB" or image.getexif() or image.info:
                    failures.append(f"Unexpected image mode or metadata: {name}")
            if (
                "sha256" in field
                and hashlib.sha256(path.read_bytes()).hexdigest()
                != field["sha256"][modality]
            ):
                failures.append(f"Hash mismatch: {name}")
        if len(set(shapes)) > 1:
            failures.append(f"Unequal modality sizes: {field['field_id']}")
    result = {
        "fields": len(records),
        "images": 3 * len(records),
        "failures": failures,
        "status": "fail" if failures else "pass",
    }
    print(json.dumps(result, indent=2))
    dump(cfg.OUTPUT_ROOT / "whole_field_verification.json", result)
    if failures:
        raise RuntimeError("Whole-field verification failed")


def prepare(extra):
    require_development("cell_manifest")
    require_development("cv_manifest")
    import cv2
    import torch
    from cellpose import models
    import segmentation as prep
    from registration import read_image, write_image, estimate_translation

    args = data_args(extra)
    verify_data(extra)
    fields = field_rows()
    if args.limit:
        fields = fields[: args.limit]
    model = models.Cellpose(gpu=torch.cuda.is_available(), model_type="cyto3")
    out = cfg.OUTPUT_ROOT / ("smoke_preparation" if args.limit else "preparation")
    (accepted, qc_all, registration) = ([], [], [])
    for n, field in enumerate(fields, 1):
        fid = field["field_id"]
        state = out / "field_records" / f"{fid}.json"
        if state.exists():
            cached = json.loads(state.read_text())
        else:
            reference = read_image(args.data / field["images"]["DFR"])
            split = {"A": "train", "B": "val", "C": "test"}[field["partition"]]
            audit = {
                "batch_folder": fid,
                "group_index": int(fid.rsplit("-", 1)[1]),
                "split": split,
                "cell_type": field["cell_type"],
            }
            for mod, key in [("DFR", "rd"), ("BFT", "rb"), ("BFR", "t")]:
                audit["raw_" + key] = str((args.data / field["images"][mod]).resolve())
                if mod != "DFR":
                    reg = estimate_translation(
                        reference, read_image(args.data / field["images"][mod])
                    )
                    if reg["method"] == "failed":
                        raise RuntimeError(f"Registration failed: {fid}/{mod}")
                    audit[key + "_registration_passed"] = reg["passed"]
                    audit[key + "_registration_consensus_px"] = reg["method_consensus_px"]
                    for key2 in ["shift_x", "shift_y", "ecc"]:
                        audit[key + "_" + key2] = reg[key2]
            (background, masked, quarantine) = [
                out / name for name in ["cells", "masked", "review"]
            ]
            rows = prep.process_group(audit, model, background, masked, quarantine)
            cells = []
            for row in rows:
                root = masked
                if row["review_reasons"] == "neighbor_contamination":
                    mask = (
                        read_image(quarantine / row["mask_path"], cv2.IMREAD_GRAYSCALE)
                        > 0
                    )
                    root = out / "neighbor_recovered"
                    for key in ["rd_path", "rb_path", "t_path"]:
                        crop = read_image(quarantine / row[key])
                        write_image(
                            root / row[key], crop * mask[..., None].astype(np.uint8)
                        )
                    write_image(root / row["mask_path"], mask.astype(np.uint8) * 255)
                elif row["status"] != "kept":
                    continue
                sid = fid + f"_cell{row['rd_label_id']:04d}"
                cell = {
                    "sample_id": sid,
                    "dataset_role": "external_test" if split == "test" else "main",
                    "original_split": split,
                    "cell_type": field["cell_type"],
                    "label": cfg.CLASS_TO_INDEX[field["cell_type"]],
                    "source_group": fid,
                    "parent_field_id": fid,
                    "image_name": sid + ".jpg",
                    "fold": field["cv_fold"],
                }
                for key in ["rd_path", "rb_path", "t_path", "mask_path"]:
                    cell[key] = (root / row[key]).resolve().as_posix()
                cells.append(cell)
            audit = {k: v for (k, v) in audit.items() if not k.startswith("raw_")}
            cached = {"cells": cells, "qc": rows, "registration": audit}
            dump(state, cached)
        accepted.extend(cached["cells"])
        qc_all.extend(cached["qc"])
        registration.append(cached["registration"])
        print(f"{n}/{len(fields)} fields; {len(accepted)} accepted cells", flush=True)
    table(out / "candidate_qc.csv", qc_all)
    table(out / "registration.csv", registration)
    target = out / "smoke_cells.csv" if args.limit else cfg.MANIFEST_PATH
    validate_cells(accepted)
    table(target, accepted)
    if not args.limit:
        main = [r for r in accepted if r["dataset_role"] == "main"]
        if any((r["fold"] is None for r in main)):
            raise RuntimeError(
                "A previously empty field now contains accepted cells. Audit segmentation before assigning its CV fold."
            )
        table(cfg.FOLD_ASSIGNMENT_PATH, main)
    dump(
        out / "summary.json",
        {
            "fields": len(fields),
            "accepted_cells": len(accepted),
            "accepted_fields": len({r["parent_field_id"] for r in accepted}),
        },
    )


def selected():
    if not cfg.SELECTED_BACKBONES_PATH.exists():
        raise FileNotFoundError("Run select after the complete six-backbone screen.")
    return json.loads(cfg.SELECTED_BACKBONES_PATH.read_text())


def train(extra, task):
    require_development("proxy_cnn" if task == "proxy-models" else "cnn")
    import cv_train_utils as trainer
    from cv_data import read_manifest
    from expanded_models import build_single_model
    from cv_models import FlexibleMultimodalModel

    p = argparse.ArgumentParser()
    p.add_argument(
        "--modalities",
        nargs="+",
        choices=list(cfg.MODALITIES),
        default=list(cfg.MODALITIES),
    )
    p.add_argument(
        "--backbones",
        nargs="+",
        choices=list(cfg.BACKBONES),
        default=list(cfg.BACKBONES),
    )
    p.add_argument("--folds", nargs="+", type=int, choices=[0, 1, 2], default=[0, 1, 2])
    args = p.parse_args(extra)
    specs = []
    if task == "screen":
        for mod in args.modalities:
            for backbone in args.backbones:
                for fold in args.folds:
                    specs.append(
                        (
                            (mod,),
                            {mod: backbone},
                            fold,
                            cfg.OUTPUT_ROOT
                            / "cnn/single"
                            / mod
                            / backbone
                            / f"fold_{fold}",
                        )
                    )
    elif task == "fusion":
        backbones = selected()
        for n in [2, 3]:
            for combo in itertools.combinations(list(cfg.MODALITIES), n):
                for fold in args.folds:
                    specs.append(
                        (
                            combo,
                            {m: backbones[m] for m in combo},
                            fold,
                            cfg.OUTPUT_ROOT
                            / "cnn/fusion"
                            / "+".join(combo)
                            / f"fold_{fold}",
                        )
                    )
    else:
        cfg.SEED = cfg.PROTOCOL["proxy_cnn_seed"]
        rows = read_manifest()
        trainer.rows_for_fold = lambda fold, train: [
            r for r in rows if r["original_split"] == ("train" if train else "val")
        ]
        for mod in args.modalities:
            backbone = cfg.LOCKED_BACKBONES[mod]
            specs.append(
                ((mod,), {mod: backbone}, 0, cfg.OUTPUT_ROOT / "proxy_cnn" / mod)
            )
    for mods, backbones, fold, out in specs:

        def factory(mods=mods, backbones=backbones):
            if len(mods) == 1:
                return build_single_model(mods[0], backbones[mods[0]])
            return FlexibleMultimodalModel(
                mods,
                {m: cfg.BACKBONES[b]["encoder_name"] for (m, b) in backbones.items()},
            )

        trainer.train_fold(
            "+".join(mods), fold, mods, factory, out
        )
        dump(
            out / "architecture.json",
            {
                "modalities": list(mods),
                "backbones": backbones,
                "fold": fold,
                "protocol": "A/B" if task == "proxy-models" else "grouped_cv",
                "modality_columns": {"BFR": "t_path", "BFT": "rb_path", "DFR": "rd_path"},
            },
        )


def select(extra):
    require_development("cnn")
    import pandas as pd

    rows = []
    for mod in cfg.MODALITIES:
        for backbone in cfg.BACKBONES:
            scores = []
            auc = []
            for fold in range(3):
                root = cfg.OUTPUT_ROOT / "cnn/single" / mod / backbone / f"fold_{fold}"
                qa = json.loads((root / "checkpoint_reload_qa.json").read_text())
                if qa["status"] != "pass":
                    raise RuntimeError(str(root))
                if qa.get("checkpoint_sha256") != sha256(root / "best_model_state.pt"):
                    raise RuntimeError(f"Checkpoint changed after validation: {root}")
                architecture = json.loads((root / "architecture.json").read_text())
                if architecture.get("modality_columns") != MODALITY_COLUMNS or architecture["backbones"] != {mod: backbone} or architecture["modalities"] != [mod]:
                    raise ValueError(f"Backbone/modality identity mismatch: {root}")
                result = json.loads((root / "experiment_result.json").read_text())
                scores.append(result["val_overall"]["macro_f1"])
                auc.append(result["val_overall"]["macro_auc_ovr"])
            if not np.isfinite(scores).all() or not np.isfinite(auc).all():
                raise ValueError(f"Nonfinite validation metrics: {mod}/{backbone}")
            rows.append(
                {
                    "modality": mod,
                    "backbone": backbone,
                    "validation_macro_f1_mean": np.mean(scores),
                    "validation_macro_f1_sd": np.std(scores, ddof=1),
                    "validation_auc_mean": np.mean(auc),
                }
            )
    frame = pd.DataFrame(rows).sort_values(
        ["modality", "validation_macro_f1_mean", "validation_auc_mean"],
        ascending=[True, False, False],
    )
    cfg.SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cfg.SUMMARY_DIR / "backbone_validation.csv", index=False)
    winners = {m: frame[frame.modality.eq(m)].iloc[0].backbone for m in cfg.MODALITIES}
    if cfg.SELECTED_BACKBONES_PATH.exists():
        if json.loads(cfg.SELECTED_BACKBONES_PATH.read_text()) != winners:
            raise RuntimeError("Validation selection changed. Use a new output directory.")
    dump(cfg.SELECTED_BACKBONES_PATH, winners)
    print(json.dumps(winners, indent=2))


def dispatch(task, extra):
    bind_run()
    from reproducibility import configure
    configure(cfg.SEED)
    if task == "verify-data":
        return verify_data(extra)
    if task == "prepare":
        return prepare(extra)
    if task in ["screen", "fusion", "proxy-models"]:
        return train(extra, task)
    if task == "select":
        return select(extra)
    if task == "evaluate":
        import evaluation

        return evaluation.main(extra)
    if task in ["proxy-features", "proxy-classify", "proxy-evaluate"]:
        import proxy_classification

        return proxy_classification.main(task, extra)
    if task == "intervene":
        import intervention

        return intervention.main(extra)
    if task == "statistics":
        import inference

        return inference.main(extra)
    import interpretation

    return interpretation.main(task, extra)
