from __future__ import annotations
import math
import cv2
import numpy as np
import intensity_kernels as base

EPS = 1e-06


def descriptor_names(code: str) -> list[str]:
    if code == "p08":
        return [
            f"depth_{layer:02d}_{band}_energy"
            for layer in range(12)
            for band in ("low", "mid", "high")
        ]
    if code == "p12":
        return ["interface_a_gradient_energy", "interface_b_gradient_energy"]
    raise KeyError(code)


def _active(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    mask = (np.asarray(mask) > 0.5).astype(np.float32)
    active = mask > 0.5
    if active.sum() < 16:
        raise ValueError("mask contains fewer than 16 pixels")
    deq = base._geometry(mask)[3]
    return (mask, active, deq)


def _depth(mask: np.ndarray, deq: float) -> np.ndarray:
    return base._depth_bins_with_guard(mask, deq, 12)


def _mean_profile(
    values: np.ndarray, groups: np.ndarray, active: np.ndarray
) -> np.ndarray:
    fallback = float(np.mean(values[active]))
    return np.asarray(
        [
            (
                np.mean(values[(groups == index) & active])
                if np.count_nonzero((groups == index) & active) >= 5
                else fallback
            )
            for index in range(12)
        ],
        dtype=np.float32,
    )


def _bands(
    y: np.ndarray, mask: np.ndarray, deq: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active = mask > 0.5
    low = base._blur_masked(y, mask, base._scaled_sigma(deq, 0.1, 2.0))
    low = low - float(np.mean(low[active]))
    mid = base._blur_masked(
        y, mask, base._scaled_sigma(deq, 0.03, 1.5)
    ) - base._blur_masked(y, mask, base._scaled_sigma(deq, 0.1, 2.0))
    high = y - base._blur_masked(y, mask, base._scaled_sigma(deq, 0.02, 1.25))
    return (low, mid, high)


def _gradient_energy(values: np.ndarray, mask: np.ndarray, deq: float) -> np.ndarray:
    resolved = base._blur_masked(values, mask, base._scaled_sigma(deq, 0.02, 1.25))
    (gy, gx) = np.gradient(resolved)
    return gx * gx + gy * gy


def _interface_bands(
    mask: np.ndarray, active: np.ndarray, deq: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inward = cv2.distanceTransform(active.astype(np.uint8), cv2.DIST_L2, 5)
    outside = cv2.distanceTransform((~active).astype(np.uint8), cv2.DIST_L2, 5)
    width = max(0.1 * deq, 2.0)
    inner = active & (inward <= width)
    outer = (outside >= 0.05 * deq) & (outside <= 0.15 * deq)
    if inner.sum() < 8 or outer.sum() < 8:
        raise ValueError("insufficient local interface pixels")
    taper = np.zeros(mask.shape, np.float32)
    taper[active] = np.clip(1.0 - inward[active] / width, 0.0, 1.0)
    return (inner, outer, taper)


def extract_descriptor(image: np.ndarray, mask: np.ndarray, code: str) -> np.ndarray:
    image = np.asarray(image, np.float32).clip(0, 1)
    (mask, active, deq) = _active(mask)
    groups = _depth(mask, deq)
    (y, a, b) = base._opponent(image)
    if code == "p08":
        rows = []
        for band in _bands(y, mask, deq):
            energy = band * band
            whole = max(float(np.mean(energy[active])), EPS)
            rows.append(_mean_profile(energy, groups, active) / whole)
        out = np.stack(rows, axis=1).reshape(-1)
    elif code == "p12":
        (inner, _, _) = _interface_bands(mask, active, deq)
        out = np.asarray(
            [
                np.mean(_gradient_energy(a, mask, deq)[inner]),
                np.mean(_gradient_energy(b, mask, deq)[inner]),
            ],
            np.float32,
        )
    else:
        raise KeyError(code)
    expected = {"p08": 36, "p12": 2}[code]
    if out.shape != (expected,) or not np.isfinite(out).all():
        raise ValueError(
            f"invalid {code} descriptor: shape={out.shape}, finite={np.isfinite(out).all()}"
        )
    return out.astype(np.float32)


def _gain_map(
    values: np.ndarray, groups: np.ndarray, active: np.ndarray, squared: bool = True
) -> np.ndarray:
    statistic = values * values if squared else np.abs(values)
    target = max(float(np.mean(statistic[active])), EPS)
    gains = np.ones(values.shape, np.float32)
    for layer in range(12):
        locations = (groups == layer) & active
        if locations.sum() >= 5:
            current = max(float(np.mean(statistic[locations])), EPS)
            gain = math.sqrt(target / current) if squared else target / current
            gains[locations] = float(np.clip(gain, 0.25, 4.0))
    return gains


def _restore_luminance_mean(
    original: np.ndarray, modified: np.ndarray, active: np.ndarray
) -> np.ndarray:
    out = modified.copy()
    out[active] += float(np.mean(original[active]) - np.mean(out[active]))
    return out


def _restore_chroma(
    original_a: np.ndarray,
    original_b: np.ndarray,
    new_a: np.ndarray,
    new_b: np.ndarray,
    active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    (mean_a, mean_b) = (
        float(np.mean(original_a[active])),
        float(np.mean(original_b[active])),
    )
    a = new_a.copy()
    b = new_b.copy()
    a[active] -= float(np.mean(a[active]))
    b[active] -= float(np.mean(b[active]))
    original_energy = float(
        np.mean((original_a[active] - mean_a) ** 2 + (original_b[active] - mean_b) ** 2)
    )
    new_energy = float(np.mean(a[active] ** 2 + b[active] ** 2))
    scale = (
        math.sqrt(original_energy / max(new_energy, EPS))
        if original_energy > EPS
        else 1.0
    )
    a[active] = a[active] * scale + mean_a
    b[active] = b[active] * scale + mean_b
    return (a, b)


def neutralise_full(
    image: np.ndarray, mask: np.ndarray, code: str, seed: int = 0
) -> tuple[np.ndarray, dict]:
    del seed
    image = np.asarray(image, np.float32).clip(0, 1)
    qc = {"proxy": code, "status": "pass", "reason": ""}
    try:
        (mask, active, deq) = _active(mask)
        groups = _depth(mask, deq)
        (y, a, b) = base._opponent(image)
        if code == "p08":
            changes = []
            for band in _bands(y, mask, deq):
                changes.append(
                    (_gain_map(band, groups, active, squared=True) - 1.0) * band
                )
            new_y = _restore_luminance_mean(y, y + sum(changes), active)
            out = base._from_opponent(new_y, a, b)
        elif code == "p12":
            (_, _, taper) = _interface_bands(mask, active, deq)
            smooth_a = base._blur_masked(a, mask, base._scaled_sigma(deq, 0.03, 1.5))
            smooth_b = base._blur_masked(b, mask, base._scaled_sigma(deq, 0.03, 1.5))
            new_a = a * (1.0 - taper) + smooth_a * taper
            new_b = b * (1.0 - taper) + smooth_b * taper
            (new_a, new_b) = _restore_chroma(a, b, new_a, new_b, active)
            out = base._from_opponent(y, new_a, new_b)
        else:
            raise KeyError(code)
        out[~active] = image[~active]
        return (np.clip(out, 0, 1).astype(np.float32), qc)
    except Exception as exc:
        qc.update(status="failed", reason=str(exc))
        return (image.copy(), qc)
