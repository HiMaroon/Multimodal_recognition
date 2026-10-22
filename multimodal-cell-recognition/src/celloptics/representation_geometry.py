import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score
import cv_config as cfg

SEED = cfg.PROTOCOL["representation_seed"]


def balanced_indices(labels: np.ndarray, n_per_class: int = 300) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    selected = []
    for label in range(3):
        candidates = np.flatnonzero(labels == label)
        selected.append(
            np.sort(
                rng.choice(
                    candidates, size=min(n_per_class, len(candidates)), replace=False
                )
            )
        )
    return np.concatenate(selected)


def embed(features: np.ndarray, labels: np.ndarray):
    scaled = StandardScaler().fit_transform(features)
    n_components = min(50, scaled.shape[0] - 1, scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=SEED, svd_solver="randomized")
    reduced = pca.fit_transform(scaled)
    kwargs = dict(
        n_components=2,
        perplexity=35,
        init="pca",
        learning_rate="auto",
        random_state=SEED,
    )
    try:
        coordinates = TSNE(max_iter=1500, **kwargs).fit_transform(reduced)
    except TypeError:
        coordinates = TSNE(n_iter=1500, **kwargs).fit_transform(reduced)
    centers = np.stack([reduced[labels == label].mean(axis=0) for label in range(3)])
    between = np.mean(
        [
            np.sum((centers[i] - centers[j]) ** 2)
            for i in range(3)
            for j in range(i + 1, 3)
        ]
    )
    within = np.mean(
        [
            np.mean(np.sum((reduced[labels == label] - centers[label]) ** 2, axis=1))
            for label in range(3)
        ]
    )
    metrics = {
        "silhouette_pca50": float(silhouette_score(reduced, labels)),
        "between_within_ratio": float(between / max(within, 1e-12)),
        "tsne_trustworthiness": float(
            trustworthiness(reduced, coordinates, n_neighbors=10)
        ),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "n": int(len(labels)),
    }
    return (coordinates, metrics)
