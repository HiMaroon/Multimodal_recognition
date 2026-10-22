from __future__ import annotations
import math
import cv2
import numpy as np

EPS = 1e-06


def luminance(image: np.ndarray) -> np.ndarray:
    return image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114


def _blur_masked(values: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 0.5)
    numerator = cv2.GaussianBlur(
        values * mask[..., None] if values.ndim == 3 else values * mask,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_CONSTANT,
    )
    denominator = cv2.GaussianBlur(
        mask.astype(np.float32),
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_CONSTANT,
    )
    if values.ndim == 3:
        denominator = denominator[..., None]
    return numerator / np.maximum(denominator, EPS)


def _scaled_sigma(deq: float, fraction: float, floor_pixels: float) -> float:
    return max(fraction * deq, floor_pixels)


def _geometry(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    (yy, xx) = np.indices(mask.shape, dtype=np.float32)
    active = mask > 0.5
    if active.sum() < 16:
        raise ValueError("mask contains fewer than 16 pixels")
    (cy, cx) = np.mean(np.where(active), axis=1)
    radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    q = max(float(np.quantile(radius[active], 0.99)), 1.0)
    rho = np.clip(radius / q, 0, 1)
    depth = cv2.distanceTransform(active.astype(np.uint8), cv2.DIST_L2, 5)
    dq = max(float(np.quantile(depth[active], 0.99)), 1.0)
    eta = np.clip(depth / dq, 0, 1)
    deq = 2 * math.sqrt(float(active.sum()) / math.pi)
    return (rho, eta, radius, deq)


def _bins(values: np.ndarray, mask: np.ndarray, n_bins: int) -> np.ndarray:
    active = mask > 0.5
    index = np.minimum((np.clip(values, 0, 1) * n_bins).astype(int), n_bins - 1)
    index[~active] = -1
    return index


def _depth_bins_with_guard(mask: np.ndarray, deq: float, n_bins: int) -> np.ndarray:
    active = mask > 0.5
    inward = cv2.distanceTransform(active.astype(np.uint8), cv2.DIST_L2, 5)
    guard = max(0.03 * deq, 1.5)
    reference = max(float(np.quantile(inward[active], 0.99)), guard + 1.0)
    eta = np.clip((inward - guard) / (reference - guard), 0, 1)
    index = _bins(eta, mask, n_bins)
    index[active & (inward < guard)] = -1
    return index


def _bin_baseline(
    values: np.ndarray, index: np.ndarray, mask: np.ndarray, n_bins: int
) -> np.ndarray:
    active = mask > 0.5
    if values.ndim == 2:
        out = values.copy()
        fallback = np.median(values[active])
        for k in range(n_bins):
            loc = index == k
            out[loc] = np.median(values[loc]) if loc.sum() >= 5 else fallback
        return out
    out = values.copy()
    fallback = np.median(values[active], axis=0)
    for k in range(n_bins):
        loc = index == k
        out[loc] = np.median(values[loc], axis=0) if loc.sum() >= 5 else fallback
    return out


def _opponent(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = luminance(image)
    c = image - y[..., None]
    a = (c[..., 0] - c[..., 1]) / math.sqrt(2)
    b = (c[..., 0] + c[..., 1] - 2 * c[..., 2]) / math.sqrt(6)
    return (y, a, b)


def _from_opponent(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    c0 = a / math.sqrt(2) + b / math.sqrt(6)
    c1 = -a / math.sqrt(2) + b / math.sqrt(6)
    c2 = -2 * b / math.sqrt(6)
    return np.stack((y + c0, y + c1, y + c2), axis=-1)


def _fourier_filter(
    y: np.ndarray, mask: np.ndarray, mode: str, deq: float
) -> np.ndarray:
    active = mask > 0.5
    inward = cv2.distanceTransform(active.astype(np.uint8), cv2.DIST_L2, 5)
    transition = max(0.08 * deq, 2.0)
    t = np.clip(inward / transition, 0, 1)
    window = (0.5 - 0.5 * np.cos(np.pi * t)) * mask
    centered = (y - np.mean(y[active])) * window
    fft = np.fft.fftshift(np.fft.fft2(centered))
    (h, w) = y.shape
    (fy, fx) = np.indices((h, w), dtype=float)
    rr = np.sqrt((fy - h / 2) ** 2 + (fx - w / 2) ** 2)
    theta = np.arctan2(fy - h / 2, fx - w / 2)
    (amp, phase) = (np.abs(fft), np.angle(fft))
    if mode == "radial":
        b = np.minimum((rr / max(rr.max(), 1) * 24).astype(int), 23)
        means = np.array([amp[b == k].mean() for k in range(24)])
        uniform = np.full_like(means, means.mean())
        target = uniform[b]
    else:
        b = np.minimum(((theta + np.pi) / (2 * np.pi) * 18).astype(int), 17)
        radial_b = np.minimum((rr / max(rr.max(), 1) * 24).astype(int), 23)
        target = amp.copy()
        for k in range(24):
            loc = radial_b == k
            target[loc] = amp[loc].mean()
    rebuilt = np.real(np.fft.ifft2(np.fft.ifftshift(target * np.exp(1j * phase))))
    difference = rebuilt - centered
    out = y.copy()
    out[active] = y[active] + (window * difference)[active]
    return out


def neutralise_full(
    image: np.ndarray, mask: np.ndarray, code: str, seed: int = 0
) -> tuple[np.ndarray, dict]:
    image = np.asarray(image, np.float32).clip(0, 1)
    mask = (np.asarray(mask) > 0.5).astype(np.float32)
    active = mask > 0.5
    qc = {"status": "pass", "reason": "", "proxy": code}
    try:
        (rho, eta, radius, deq) = _geometry(mask)
        (y, a, b) = _opponent(image)
        radial12 = _bins(rho, mask, 12)
        depth12 = _depth_bins_with_guard(mask, deq, 12)
        radial6 = _bins(rho, mask, 6)
        out = image.copy()
        if code == "p01":
            out = _from_opponent(y - np.median(y[active]) * mask, a, b)
        elif code == "p04":
            base = _bin_baseline(y, radial12, mask, 12)
            out = _from_opponent(y - (base - np.median(y[active])), a, b)
        elif code == "p07":
            base = _bin_baseline(y, depth12, mask, 12)
            out = _from_opponent(y - (base - np.median(y[active])), a, b)
        elif code == "p13":
            base = _bin_baseline(y, radial12, mask, 12)
            residual = y - base
            out = _from_opponent(
                y - _blur_masked(residual, mask, _scaled_sigma(deq, 0.04, 1.5)), a, b
            )
        elif code == "p11":
            smooth = _blur_masked(y, mask, _scaled_sigma(deq, 0.02, 1.5))
            out = _from_opponent(smooth, a, b)
        elif code == "p02":
            out = _from_opponent(_fourier_filter(y, mask, "radial", deq), a, b)
        elif code == "p14":
            out = _from_opponent(_fourier_filter(y, mask, "orientation", deq), a, b)
        elif code == "p03":
            out = _from_opponent(y, a - np.mean(a[active]), b - np.mean(b[active]))
        elif code == "p06":
            (ba, bb) = (
                _bin_baseline(a, radial6, mask, 6),
                _bin_baseline(b, radial6, mask, 6),
            )
            out = _from_opponent(
                y, a - (ba - np.mean(a[active])), b - (bb - np.mean(b[active]))
            )
        elif code == "p09":
            (ba, bb) = (
                _bin_baseline(a, depth12, mask, 12),
                _bin_baseline(b, depth12, mask, 12),
            )
            out = _from_opponent(
                y, a - (ba - np.mean(a[active])), b - (bb - np.mean(b[active]))
            )
        else:
            raise KeyError(code)
        out[~active] = image[~active]
        return (np.clip(out, 0, 1).astype(np.float32), qc)
    except Exception as exc:
        qc.update(status="failed", reason=str(exc))
        return (image.copy(), qc)
