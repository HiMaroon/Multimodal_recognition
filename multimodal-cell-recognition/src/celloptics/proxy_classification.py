import argparse
import itertools
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
import cv_config as cfg
from cv_data import ManifestDataset, BatchSpec, read_manifest
from evaluation import metrics
from workflow import dump
import proxies
from protocol_guard import require_development, freeze

SEED = cfg.PROTOCOL["proxy_classifier_seed"]
MODS = ["BFR", "BFT", "DFR"]
OUT = cfg.OUTPUT_ROOT / "proxy_classification"


def candidates():
    result = [{"family": "logistic", "C": c} for c in [0.01, 0.1, 1, 10]]
    result += [
        {"family": "rbf_svm", "C": c, "gamma_factor": g}
        for c in [1, 10]
        for g in [1, 0.1]
    ]
    result += [{"family": "lda", "shrinkage": s} for s in ["auto", 0.1]]
    result += [{"family": "extra_trees", "max_depth": d} for d in [None, 10]]
    return result


def estimator(config, dimension):
    family = config["family"]
    if family == "logistic":
        clf = LogisticRegression(
            C=config["C"], max_iter=4000, class_weight="balanced", random_state=SEED
        )
    elif family == "rbf_svm":
        clf = SVC(
            C=config["C"],
            gamma=config["gamma_factor"] / dimension,
            probability=True,
            class_weight="balanced",
            random_state=SEED,
            cache_size=512,
        )
    elif family == "lda":
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=config["shrinkage"])
    else:
        clf = ExtraTreesClassifier(
            n_estimators=250,
            max_depth=config["max_depth"],
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=1,
        )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", clf),
        ]
    )


def features(limit=0, splits=("train", "val")):
    rows = read_manifest()
    for mod in MODS:
        for split in splits:
            subset = [r for r in rows if r["original_split"] == split]
            if limit:
                subset = subset[:limit]
            target = (
                OUT / ("smoke_features" if limit else "features") / f"{mod}_{split}.npz"
            )
            if target.exists():
                continue
            dataset = ManifestDataset(subset, BatchSpec((mod,), False))
            values = {r["id"]: [] for r in proxies.REGISTRY}
            for i in range(len(dataset)):
                sample = dataset[i]
                image = np.moveaxis(sample["images"].numpy(), 0, -1)
                mask = sample["mask"][0].numpy() > 0.5
                for r in proxies.REGISTRY:
                    vector = proxies.descriptor(image, mask, r["id"])
                    if len(vector) != r["dimension"]:
                        raise ValueError(r["id"])
                    values[r["id"]].append(vector)
                if i % 200 == 0:
                    print(f"{mod}/{split} {i}/{len(dataset)}", flush=True)
            payload = {k: np.asarray(v, np.float32) for (k, v) in values.items()}
            payload.update(
                {
                    k: np.asarray([r[k] for r in subset])
                    for k in ["sample_id", "parent_field_id", "cell_type"]
                }
            )
            payload["label"] = np.asarray([int(r["label"]) for r in subset])
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.with_suffix(".tmp").open("wb") as f:
                np.savez_compressed(f, **payload)
            target.with_suffix(".tmp").replace(target)


def load_features(splits):
    data = {}
    rows = read_manifest()
    for m in MODS:
        for s in splits:
            with np.load(OUT / "features" / f"{m}_{s}.npz", allow_pickle=False) as z:
                data[m, s] = {k: z[k] for k in z.files}
            expected = [r for r in rows if r["original_split"] == s]
            for key in ["sample_id", "parent_field_id", "cell_type", "label"]:
                values = np.asarray([r[key] for r in expected], dtype=int if key == "label" else str)
                if not np.array_equal(values, data[m, s][key]):
                    raise ValueError(f"Feature/manifest mismatch: {m}/{s}/{key}")
            for item in proxies.REGISTRY:
                if data[m, s][item["id"]].shape != (len(expected), item["dimension"]):
                    raise ValueError(f"Incorrect descriptor dimensions: {m}/{s}/{item['id']}")
    for s in splits:
        for m in MODS[1:]:
            if not np.array_equal(data["BFR", s]["sample_id"], data[m, s]["sample_id"]):
                raise ValueError("Modalities are not aligned")
    return data


def combinations():
    return [c for count in (1, 2, 3) for c in itertools.combinations(MODS, count)]


def classify():
    require_development("proxy_classifier")
    data = load_features(("train", "val"))
    keys = list(proxies.BY_ID)
    for combo in combinations():
        name = "+".join(combo)
        directory = OUT / "models" / name
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "selection.json").exists():
            if not (directory / "model.joblib").exists():
                raise RuntimeError(f"Missing selected model: {directory}")
            continue
        x = {
            s: np.concatenate([data[m, s][k] for m in combo for k in keys], axis=1)
            for s in ["train", "val"]
        }
        y = {s: data["BFR", s]["label"] for s in x}
        trials = []
        best = -1
        with threadpool_limits(limits=1):
            for index, config in enumerate(candidates()):
                model = estimator(config, x["train"].shape[1])
                model.fit(x["train"], y["train"])
                score = f1_score(y["val"], model.predict(x["val"]), average="macro")
                trials.append(
                    {
                        "candidate": index,
                        "validation_macro_f1": score,
                        "configuration": config,
                    }
                )
                if score > best + 1e-12:
                    best = score
                    chosen = model
                    chosen_config = config
            joblib.dump(chosen, directory / "model.joblib")
            dump(directory / "validation_trials.json", trials)
            dump(
                directory / "selection.json",
                {"validation_macro_f1": best, "configuration": chosen_config,
                 "training_partition": "A", "selection_partition": "B",
                 "modalities": list(combo), "proxies": keys},
            )


def evaluate_selected():
    combos = combinations()
    files = [OUT / "models" / "+".join(c) / n for c in combos
             for n in ("model.joblib", "selection.json", "validation_trials.json")]
    files += [OUT / "features" / f"{m}_{s}.npz" for m in MODS for s in ("train", "val")]
    freeze("proxy_classifier", files + [cfg.MANIFEST_PATH])
    features(splits=("test",))
    data = load_features(("test",))
    keys = list(proxies.BY_ID)
    records = []
    for combo in combos:
        name = "+".join(combo)
        directory = OUT / "models" / name
        if (directory / "metrics.json").exists():
            records.append(json.loads((directory / "metrics.json").read_text()))
            continue
        selection = json.loads((directory / "selection.json").read_text())
        chosen = joblib.load(directory / "model.joblib")
        x = {"test": np.concatenate([data[m, "test"][k] for m in combo for k in keys], axis=1)}
        y = {"test": data["BFR", "test"]["label"]}
        with threadpool_limits(limits=1):
            probability = chosen.predict_proba(x["test"])
            prediction = chosen.predict(x["test"])
        result = {
            "modalities": name,
            "dimension": x["test"].shape[1],
            "validation_macro_f1": selection["validation_macro_f1"],
            **metrics(y["test"], probability),
        }
        from sklearn.metrics import precision_score, recall_score, confusion_matrix

        result.update(
            macro_f1=f1_score(y["test"], prediction, average="macro"),
            macro_precision=precision_score(
                y["test"], prediction, average="macro", zero_division=0
            ),
            macro_recall=recall_score(
                y["test"], prediction, average="macro", zero_division=0
            ),
            n_errors=int(np.sum(prediction != y["test"])),
            confusion_matrix=confusion_matrix(
                y["test"], prediction, labels=[0, 1, 2]
            ).tolist(),
        )
        tab = pd.DataFrame(
            {
                k: data["BFR", "test"][k]
                for k in ["sample_id", "parent_field_id", "cell_type", "label"]
            }
        )
        tab["prediction"] = prediction
        for j in range(3):
            tab[f"prob_{j}"] = probability[:, j]
        tab.to_csv(directory / "predictions.csv", index=False)
        dump(directory / "metrics.json", result)
        records.append(result)
    dump(OUT / "classification_metrics.json", records)


def main(task, extra):
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args(extra)
    if task == "proxy-features":
        require_development("proxy_classifier")
        features(args.limit)
    elif task == "proxy-classify":
        classify()
    else:
        evaluate_selected()
