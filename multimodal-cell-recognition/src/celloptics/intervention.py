import argparse
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import cv_config as cfg
from cv_data import read_manifest, BatchSpec, ManifestDataset
from evaluation import load_model, predict
from workflow import dump
import proxies
from protocol_guard import freeze, model_files, sha256


def main(extra):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--modalities",
        nargs="+",
        choices=list(cfg.MODALITIES),
        default=list(cfg.MODALITIES),
    )
    p.add_argument(
        "--proxies", nargs="+", choices=list(proxies.BY_ID), default=list(proxies.BY_ID)
    )
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args(extra)
    freeze("proxy_cnn", model_files([cfg.OUTPUT_ROOT / "proxy_cnn" / m for m in cfg.MODALITIES]) + [cfg.MANIFEST_PATH])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [r for r in read_manifest() if r["original_split"] == "test"]
    lookup = {r["sample_id"]: r for r in rows}
    for mod in args.modalities:
        model, model_modalities = load_model(cfg.OUTPUT_ROOT / "proxy_cnn" / mod, device)
        if model_modalities != (mod,):
            raise ValueError(f"Model/input modality mismatch: {mod}")
        checkpoint_hash = sha256(cfg.OUTPUT_ROOT / "proxy_cnn" / mod / "best_model_state.pt")
        dataset = ManifestDataset(rows, BatchSpec((mod,), False))
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=1)
        baseline_path = cfg.OUTPUT_ROOT / "interventions" / f"baseline_{mod}.npz"
        if baseline_path.exists():
            z = np.load(baseline_path, allow_pickle=False)
            if list(z["sample_id"]) != [r["sample_id"] for r in rows]:
                raise ValueError("Baseline identity mismatch")
            if str(z["checkpoint_sha256"]) != checkpoint_hash:
                raise ValueError("Baseline checkpoint mismatch")
            baseline = z["probabilities"]
        else:
            baseline = np.concatenate(
                [predict(model, b["images"].numpy(), device) for b in loader]
            )
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                baseline_path,
                sample_id=[r["sample_id"] for r in rows],
                probabilities=baseline,
                checkpoint_sha256=np.asarray(checkpoint_hash),
            )
        base_by_id = dict(zip([r["sample_id"] for r in rows], baseline))
        for code in args.proxies:
            out = cfg.OUTPUT_ROOT / "interventions" / f"{mod}_{code}"
            marker = out / "metadata.json"
            if (
                marker.exists()
                and json.loads(marker.read_text()).get("status") == "complete"
            ):
                continue
            records = []
            checks = []
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for batch_no, batch in enumerate(loader, 1):
                    chw = batch["images"].numpy()
                    mask = batch["mask"][:, 0].numpy() > 0.5
                    ids = list(batch["sample_id"])
                    labels = batch["label"].numpy()
                    originals = [np.moveaxis(x, 0, -1) for x in chw]
                    full = [
                        proxies.neutralize(im, m, code, 260830000 + batch_no * 1000 + i)
                        for (i, (im, m)) in enumerate(zip(originals, mask))
                    ]
                    for dose in cfg.PROTOCOL["intervention_doses"]:
                        physical = np.asarray(
                            [
                                np.moveaxis(
                                    np.clip(im + dose * (v[0] - im), 0, 1), -1, 0
                                )
                                for (im, v) in zip(originals, full)
                            ],
                            np.float32,
                        )
                        pp = predict(model, physical, device)
                        count = cfg.PROTOCOL["matched_controls"]
                        jobs = [(i, r) for i in range(len(ids)) for r in range(count)]

                        def make(job):
                            (i, r) = job
                            return proxies.control(
                                originals[i],
                                full[i][0],
                                mask[i],
                                dose,
                                proxies.control_seed(code, ids[i], dose, r),
                                code,
                            )

                        controls = list(pool.map(make, jobs))
                        cp = (
                            predict(
                                model,
                                np.asarray(
                                    [np.moveaxis(v[0], -1, 0) for v in controls],
                                    np.float32,
                                ),
                                device,
                            )
                            .reshape(len(ids), count, 3)
                            .mean(1)
                        )
                        for i, sid in enumerate(ids):
                            y = labels[i]
                            bp = base_by_id[sid][y]
                            records.append(
                                {
                                    "sample_id": sid,
                                    "parent_field_id": lookup[sid]["parent_field_id"],
                                    "label": int(y),
                                    "modality": mod,
                                    "proxy_id": code,
                                    "dose": dose,
                                    "baseline_true_probability": float(bp),
                                    "physical_true_probability": float(pp[i, y]),
                                    "control_mean_true_probability": float(cp[i, y]),
                                    "physical_loss": float(bp - pp[i, y]),
                                    "control_loss": float(bp - cp[i, y]),
                                    "specificity": float(cp[i, y] - pp[i, y]),
                                }
                            )
                            for r in range(count):
                                checks.append(
                                    {
                                        "sample_id": sid,
                                        "dose": dose,
                                        "replicate": r,
                                        "proxy_status": full[i][1]["status"],
                                        **controls[i * count + r][1],
                                    }
                                )
                    print(f"{mod}/{code} batch {batch_no}/{len(loader)}", flush=True)
            out.mkdir(parents=True, exist_ok=True)
            for name, values in [("cell_level", records), ("qc", checks)]:
                tmp = out / f"{name}.tmp"
                pd.DataFrame(values).to_csv(tmp, index=False)
                tmp.replace(out / f"{name}.csv")
            fails = sum(
                (x["status"] != "pass" or x["proxy_status"] != "pass" for x in checks)
            )
            dump(
                marker,
                {
                    "status": "complete" if fails == 0 else "qc_failed",
                    "n_cells": len(rows),
                    "qc_failures": fails,
                    "doses": 4,
                    "controls_per_dose": 8,
                    "checkpoint_sha256": checkpoint_hash,
                },
            )
            if fails:
                raise RuntimeError(
                    f"{mod}/{code}: {fails} QC failures; inspect before inference."
                )
