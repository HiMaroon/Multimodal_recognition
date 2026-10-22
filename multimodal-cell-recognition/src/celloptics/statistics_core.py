import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def f1(cm):
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    den = cm.sum(axis=-1) + cm.sum(axis=-2)
    return np.divide(
        2 * tp, den, out=np.zeros_like(tp, dtype=float), where=den != 0
    ).mean(axis=-1)


def tail(values):
    n = len(values)
    return min(
        1.0,
        2
        * min((np.sum(values <= 0) + 1) / (n + 1), (np.sum(values >= 0) + 1) / (n + 1)),
    )


def weights(labels, n, seed):
    labels = np.asarray(labels)
    if n < 1 or labels.ndim != 1 or len(labels) == 0 or not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("Invalid bootstrap labels or replicate count")
    rng = np.random.default_rng(seed)
    w = np.zeros((n, len(labels)), int)
    for label in [0, 1, 2]:
        idx = np.flatnonzero(labels == label)
        if not len(idx):
            continue
        w[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), n)
    return w


def compare(a, b, n, seed):
    a = pd.read_csv(a)
    b = paired_rows(a, pd.read_csv(b))
    fields = sorted(a.parent_field_id.unique())
    labels = a.groupby("parent_field_id").label.first().reindex(fields).to_numpy()
    w = weights(labels, n, seed)
    scores = []
    for frame in [a, b]:
        cms = np.array(
            [
                confusion_matrix(g.label, g.prediction, labels=[0, 1, 2])
                for (_, g) in frame.groupby("parent_field_id", sort=True)
            ]
        )
        scores.append((f1(cms.sum(axis=0)), f1(np.einsum("bf,fij->bij", w, cms))))
    draws = scores[0][1] - scores[1][1]
    return {
        "difference_pp": 100 * (scores[0][0] - scores[1][0]),
        "ci_low_pp": 100 * np.quantile(draws, 0.025),
        "ci_high_pp": 100 * np.quantile(draws, 0.975),
        "p": tail(draws),
        "replicates": n,
    }


def interventions(paths, n, seed):
    records = []
    for path in paths:
        frame = pd.read_csv(path)
        if frame.duplicated(["sample_id", "dose"]).any():
            raise ValueError(f"Duplicate intervention observations: {path}")
        if not np.isfinite(frame.specificity).all():
            raise ValueError(f"Nonfinite intervention values: {path}")
        if frame.groupby("sample_id").dose.nunique().ne(4).any():
            raise ValueError(f"Incomplete dose series: {path}")
        if frame.groupby("parent_field_id").label.nunique().ne(1).any():
            raise ValueError(f"Mixed labels within parent field: {path}")
        g = (
            frame.groupby(["label", "parent_field_id", "dose"])
            .specificity.mean()
            .unstack("dose")
            .sort_index()
        )
        if set(g.columns) != {0.25, 0.5, 0.75, 1.0} or g.isna().any().any():
            raise ValueError(f"Missing or unexpected intervention dose: {path}")
        values = g[[0.25, 0.5, 0.75, 1.0]].to_numpy()
        labels = g.index.get_level_values(0).to_numpy()
        if set(labels) != {0, 1, 2}:
            raise ValueError(f"Class-macro analysis requires all three classes: {path}")
        w = weights(labels, n, seed).astype(float)
        for label in [0, 1, 2]:
            w[:, labels == label] /= 3 * np.sum(labels == label)
        point = np.mean(
            [values[labels == label].mean(axis=0) for label in [0, 1, 2]], axis=0
        )
        draws = w @ values
        for endpoint, column in [
            ("S25", 0),
            ("S50", 1),
            ("S75", 2),
            ("S100", 3),
            ("dose_auc", None),
        ]:
            estimate = (
                point[column]
                if column is not None
                else point @ np.array([0.25, 0.25, 0.25, 0.125])
            )
            dist = (
                draws[:, column]
                if column is not None
                else draws @ np.array([0.25, 0.25, 0.25, 0.125])
            )
            records.append(
                {
                    "source": str(path),
                    "endpoint": endpoint,
                    "effect_pp": 100 * estimate,
                    "ci_low_pp": 100 * np.quantile(dist, 0.025),
                    "ci_high_pp": 100 * np.quantile(dist, 0.975),
                    "p": tail(dist),
                    "n_fields": len(g),
                }
            )
    return records


def paired_rows(reference, candidate):
    for frame in (reference, candidate):
        if frame.sample_id.duplicated().any():
            raise ValueError("Duplicate sample IDs")
        if frame.groupby("parent_field_id").label.nunique().ne(1).any():
            raise ValueError("Mixed labels within a parent field")
    if set(reference.sample_id) != set(candidate.sample_id):
        raise ValueError("Paired sample sets differ")
    candidate = candidate.set_index("sample_id").loc[reference.sample_id].reset_index()
    if not np.array_equal(reference.label, candidate.label) or not np.array_equal(reference.parent_field_id, candidate.parent_field_id):
        raise ValueError("Paired label or parent-field mismatch")
    return candidate
