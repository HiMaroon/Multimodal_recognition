from __future__ import annotations
import cv2
import numpy as np
import intensity_kernels as atlas

EPS = 1e-06


def _safe_median(values: np.ndarray, fallback: float = 0.0) -> float:
    return float(np.median(values)) if values.size else float(fallback)


def _profile(
    values: np.ndarray, groups: np.ndarray, n: int, active: np.ndarray
) -> np.ndarray:
    fallback = _safe_median(values[active])
    return np.asarray(
        [_safe_median(values[(groups == k) & active], fallback) for k in range(n)],
        dtype=np.float32,
    )


def _sector_profile(values: np.ndarray, active: np.ndarray, n: int = 12) -> np.ndarray:
    (yy, xx) = np.indices(active.shape, dtype=np.float32)
    (cy, cx) = np.mean(np.where(active), axis=1)
    angle = np.arctan2(yy - cy, xx - cx)
    sector = np.minimum(((angle + np.pi) / (2 * np.pi) * n).astype(int), n - 1)
    fallback = _safe_median(values[active])
    return np.asarray(
        [_safe_median(values[(sector == k) & active], fallback) for k in range(n)],
        dtype=np.float32,
    )


def _windowed_spectrum(y: np.ndarray, mask: np.ndarray, deq: float):
    active = mask > 0.5
    inward = cv2.distanceTransform(active.astype(np.uint8), cv2.DIST_L2, 5)
    transition = max(0.08 * deq, 2.0)
    taper = np.clip(inward / transition, 0, 1)
    window = (0.5 - 0.5 * np.cos(np.pi * taper)) * mask
    centered = (y - np.mean(y[active])) * window
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
    (h, w) = y.shape
    (fy, fx) = np.indices((h, w), dtype=np.float32)
    (dy, dx) = (fy - h / 2, fx - w / 2)
    radius = np.sqrt(dy**2 + dx**2)
    angle = np.arctan2(dy, dx)
    return (power, radius, angle)


def descriptor_names(code: str) -> list[str]:
    if code == "p01":
        return ["cell_median_luminance"]
    if code in {"p04", "p11"}:
        return [f"ring_{k:02d}" for k in range(12)]
    if code == "p07":
        return [f"depth_{k:02d}" for k in range(12)]
    if code in {"p13", "p14"}:
        return [f"sector_{k:02d}" for k in range(12)]
    if code in {"p02", "p03"}:
        return [code.lower()]
    if code == "p06":
        return [f"ring_{k:02d}_{c}" for k in range(6) for c in ("a", "b")]
    if code == "p09":
        return [f"depth_{k:02d}_{c}" for k in range(12) for c in ("a", "b")]
    raise KeyError(code)


def extract_descriptor(image: np.ndarray, mask: np.ndarray, code: str) -> np.ndarray:
    image = np.asarray(image, np.float32).clip(0, 1)
    mask = (np.asarray(mask) > 0.5).astype(np.float32)
    active = mask > 0.5
    if active.sum() < 16:
        raise ValueError("mask contains fewer than 16 pixels")
    (rho, _, _, deq) = atlas._geometry(mask)
    radial12 = atlas._bins(rho, mask, 12)
    radial6 = atlas._bins(rho, mask, 6)
    depth12 = atlas._depth_bins_with_guard(mask, deq, 12)
    (y, a, b) = atlas._opponent(image)
    cell_median = _safe_median(y[active])
    if code == "p01":
        out = np.asarray([cell_median], dtype=np.float32)
    elif code == "p04":
        out = _profile(y, radial12, 12, active) - cell_median
    elif code == "p07":
        out = _profile(y, depth12, 12, active) - cell_median
    elif code == "p13":
        radial_base = atlas._bin_baseline(y, radial12, mask, 12)
        residual = np.abs(y - radial_base)
        out = _sector_profile(residual, active, 12)
    elif code == "p11":
        smooth = atlas._blur_masked(y, mask, atlas._scaled_sigma(deq, 0.02, 1.5))
        (gy, gx) = np.gradient(smooth)
        out = _profile(np.sqrt(gx * gx + gy * gy), radial12, 12, active)
    elif code in {"p02", "p14"}:
        (power, fr, fa) = _windowed_spectrum(y, mask, deq)
        valid = fr > 0
        if code == "p02":
            out = np.asarray(
                [
                    np.sum(fr[valid] * power[valid])
                    / max(np.sum(power[valid]), EPS)
                    / max(fr.max(), 1)
                ],
                dtype=np.float32,
            )
        else:
            bins = np.minimum(((fa + np.pi) / (2 * np.pi) * 12).astype(int), 11)
            energies = np.asarray(
                [power[(bins == k) & valid].sum() for k in range(12)], dtype=np.float32
            )
            out = energies / max(float(energies.sum()), EPS)
    elif code == "p03":
        out = np.asarray(
            [np.median(np.sqrt(a[active] ** 2 + b[active] ** 2))], dtype=np.float32
        )
    elif code in {"p06", "p09"}:
        (groups, n) = (radial6, 6) if code == "p06" else (depth12, 12)
        out = np.asarray(
            [
                [
                    _safe_median(a[(groups == k) & active]),
                    _safe_median(b[(groups == k) & active]),
                ]
                for k in range(n)
            ],
            dtype=np.float32,
        ).reshape(-1)
    else:
        raise KeyError(code)
    expected = descriptor_names(code)
    if out.shape != (len(expected),) or not np.all(np.isfinite(out)):
        raise ValueError(
            f"invalid descriptor {code}: shape={out.shape}, finite={np.all(np.isfinite(out))}"
        )
    return out.astype(np.float32)
