import argparse
import itertools
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import cv_config as cfg
from statistics_core import f1, weights, compare, interventions, paired_rows, tail


def adjust(p):
    p = np.asarray(p, float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("BH correction requires finite p values in [0, 1]")
    order = np.argsort(p)
    ranked = p[order]
    n = len(p)
    values = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    result = np.empty(n)
    result[order] = np.clip(values, 0, 1)
    return result


def classification(task):
    root = cfg.OUTPUT_ROOT / (
        "evaluation" if task == "cnn" else "proxy_classification/models"
    )
    names = [
        "+".join(c)
        for n in [1, 2, 3]
        for c in itertools.combinations(
            ["BFT", "BFR", "DFR"] if task == "cnn" else ["BFR", "BFT", "DFR"], n
        )
    ]
    path = lambda name: (
        root / (name + ".csv") if task == "cnn" else root / name / "predictions.csv"
    )
    n = cfg.PROTOCOL["cnn_bootstrap_replicates" if task == "cnn" else "proxy_bootstrap_replicates"]
    seed = cfg.PROTOCOL["statistics_seed"]
    summary = []
    comparisons = []
    for name in names:
        frame = pd.read_csv(path(name))
        paired_rows(frame, frame)
        groups = frame.groupby("parent_field_id", sort=True)
        labels = groups.label.first().to_numpy()
        cms = np.array(
            [
                confusion_matrix(g.label, g.prediction, labels=[0, 1, 2])
                for (_, g) in groups
            ]
        )
        w = weights(labels, n, seed)
        draws = f1(np.einsum("bf,fij->bij", w, cms))
        summary.append(
            {
                "model": name,
                "macro_f1": float(f1(cms.sum(0))),
                "ci_low": np.quantile(draws, 0.025),
                "ci_high": np.quantile(draws, 0.975),
                "replicates": n,
            }
        )
    for name in names[:-1]:
        result = compare(path(names[-1]), path(name), n, seed)
        result["p_method"] = "two_sided_parent_field_bootstrap"
        comparisons.append({"reference": names[-1], "comparator": name, **result})
    out = cfg.OUTPUT_ROOT / "statistics"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(out / f"{task}_f1_intervals.csv", index=False)
    pd.DataFrame(comparisons).to_csv(
        out / f"{task}_paired_comparisons.csv", index=False
    )


def intervention_summary():
    paths = []
    for mod in ["BFR", "BFT", "DFR"]:
        for i in range(1, 16):
            root = cfg.OUTPUT_ROOT / "interventions" / f"{mod}_p{i:02d}"
            metadata = json.loads((root / "metadata.json").read_text())
            if metadata["status"] != "complete" or metadata["qc_failures"] != 0:
                raise ValueError(f"Incomplete or failed QC: {root}")
            paths.append(root / "cell_level.csv")
    records = interventions(paths, cfg.PROTOCOL["intervention_bootstrap_replicates"], cfg.PROTOCOL["statistics_seed"])
    for r in records:
        (r["modality"], r["proxy_id"]) = (
            r.pop("source").replace("\\", "/").split("/")[-2].split("_")
        )
    frame = pd.DataFrame(records)
    auc = frame.endpoint.eq("dose_auc")
    if auc.sum() != 45:
        raise ValueError("The BH family must contain exactly 45 R tests")
    frame.loc[auc, "q_bh_45"] = adjust(frame.loc[auc, "p"].to_numpy())
    out = cfg.OUTPUT_ROOT / "statistics"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "intervention_effects.csv", index=False)


def confidence():
    root = cfg.OUTPUT_ROOT / "evaluation"
    bfr = pd.read_csv(root / "BFR.csv")
    tri = paired_rows(bfr, pd.read_csv(root / "BFT+BFR+DFR.csv"))
    confidence = bfr[["prob_0", "prob_1", "prob_2"]].max(axis=1)
    bins = pd.qcut(confidence.rank(method="first"), 3, labels=["low", "middle", "high"])
    rows = []
    for name in ["low", "middle", "high"]:
        keep = bins == name
        a = bfr.loc[keep]
        b = tri.loc[keep]
        rows.append(
            {
                "confidence_tertile": name,
                "n": int(keep.sum()),
                "bfr_errors": int((a.label != a.prediction).sum()),
                "trimodal_errors": int((b.label != b.prediction).sum()),
                "corrected": int(
                    ((a.label != a.prediction) & (b.label == b.prediction)).sum()
                ),
                "new_errors": int(
                    ((a.label == a.prediction) & (b.label != b.prediction)).sum()
                ),
            }
        )
    out = cfg.OUTPUT_ROOT / "statistics"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "bfr_confidence_complementarity.csv", index=False)
    gains = []
    for combination in ["BFT+BFR", "BFR+DFR", "BFT+BFR+DFR"]:
        fusion = paired_rows(bfr, pd.read_csv(root / f"{combination}.csv"))
        labels = bfr.label.to_numpy(dtype=int)
        baseline = bfr[[f"prob_{j}" for j in range(3)]].to_numpy()[np.arange(len(bfr)), labels]
        probability = fusion[[f"prob_{j}" for j in range(3)]].to_numpy()[np.arange(len(fusion)), labels]
        for level in ["low", "middle", "high"]:
            subset = bfr.loc[bins.eq(level), ["parent_field_id", "label"]].copy()
            subset["gain"] = (probability - baseline)[bins.eq(level)]
            fields = subset.groupby("parent_field_id", sort=True).agg(label=("label", "first"), total=("gain", "sum"), count=("gain", "size"))
            w = weights(fields.label.to_numpy(), cfg.PROTOCOL["cnn_bootstrap_replicates"], cfg.PROTOCOL["statistics_seed"])
            draws = (w @ fields.total.to_numpy()) / (w @ fields["count"].to_numpy())
            gains.append({"modalities": combination, "confidence_tertile": level,
                          "n_cells": len(subset), "n_fields": len(fields),
                          "true_class_gain_pp": 100 * subset.gain.mean(),
                          "ci_low_pp": 100 * np.quantile(draws, .025),
                          "ci_high_pp": 100 * np.quantile(draws, .975), "p": tail(draws)})
    pd.DataFrame(gains).to_csv(out / "bfr_confidence_probability_gains.csv", index=False)


def main(extra):
    p = argparse.ArgumentParser()
    p.add_argument("analysis", choices=["cnn", "proxy", "interventions", "confidence"])
    args = p.parse_args(extra)
    if args.analysis in ["cnn", "proxy"]:
        classification(args.analysis)
    elif args.analysis == "confidence":
        confidence()
    else:
        intervention_summary()
