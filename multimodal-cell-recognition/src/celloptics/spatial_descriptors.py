from __future__ import annotations
import cv2
import numpy as np
import intensity_kernels as atlas

EPS = 1e-07


def descriptor_names(code: str) -> list[str]:
    if code == "p05":
        return [
            f"ring_{r:02d}_{band}" for r in range(12) for band in ("low", "mid", "high")
        ]
    if code == "p10":
        return ["boundary_minus_interior_luminance"]
    if code == "p15":
        return [f"ring_{r:02d}_angular_chroma_variance" for r in range(6)]
    raise KeyError(code)


def _sector_map(active: np.ndarray, n: int = 12) -> np.ndarray:
    (yy, xx) = np.indices(active.shape, dtype=np.float32)
    (cy, cx) = np.mean(np.where(active), axis=1)
    angle = np.arctan2(yy - cy, xx - cx)
    return np.minimum(((angle + np.pi) / (2 * np.pi) * n).astype(int), n - 1)


def extract_single(image: np.ndarray, mask: np.ndarray, code: str) -> np.ndarray:
    image = np.asarray(image, np.float32).clip(0, 1)
    mask = (np.asarray(mask) > 0.5).astype(np.float32)
    active = mask > 0.5
    if active.sum() < 16:
        raise ValueError("mask contains fewer than 16 pixels")
    (rho, _, _, deq) = atlas._geometry(mask)
    (y, a, b) = atlas._opponent(image)
    radial12 = atlas._bins(rho, mask, 12)
    radial6 = atlas._bins(rho, mask, 6)
    if code == "p05":
        low = atlas._blur_masked(y, mask, max(0.1 * deq, 2.0))
        mid = atlas._blur_masked(y, mask, max(0.03 * deq, 1.5)) - low
        high = y - atlas._blur_masked(y, mask, max(0.02 * deq, 1.25))
        values = []
        for ring in range(12):
            loc = (radial12 == ring) & active
            values.extend(
                (
                    float(np.mean(component[loc] ** 2)) if loc.any() else 0.0
                    for component in (low, mid, high)
                )
            )
        out = np.asarray(values, np.float32)
    elif code == "p10":
        inward = cv2.distanceTransform(active.astype(np.uint8), cv2.DIST_L2, 5)
        boundary = active & (inward <= 5.0)
        interior = active & (inward > 5.0)
        if boundary.sum() < 8 or interior.sum() < 8:
            raise ValueError("insufficient boundary/interior pixels")
        out = np.asarray([np.mean(y[boundary]) - np.mean(y[interior])], np.float32)
    elif code == "p15":
        sector = _sector_map(active, 12)
        chroma = np.sqrt(a * a + b * b)
        values = []
        for ring in range(6):
            medians = [
                np.median(chroma[(radial6 == ring) & (sector == s) & active])
                for s in range(12)
                if np.any((radial6 == ring) & (sector == s) & active)
            ]
            values.append(float(np.var(medians)) if len(medians) >= 3 else 0.0)
        out = np.asarray(values, np.float32)
    else:
        raise KeyError(code)
    if out.shape != (len(descriptor_names(code)),) or not np.all(np.isfinite(out)):
        raise ValueError(f"invalid {code}: {out}")
    return out
