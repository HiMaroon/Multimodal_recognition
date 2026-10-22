from __future__ import annotations
import cv2
import numpy as np
import intensity_kernels as atlas
from spatial_descriptors import _sector_map

EPS = 1e-07


def _replace_luminance(image, new_y):
    old = np.mean(image, axis=2)
    delta = (new_y - old)[..., None]
    return np.clip(image + delta, 0, 1).astype(np.float32)


def neutralise_full(image, mask, code):
    image = np.asarray(image, np.float32).clip(0, 1)
    mask = (np.asarray(mask) > 0.5).astype(np.float32)
    active = mask > 0.5
    (rho, _, _, deq) = atlas._geometry(mask)
    (y, a, b) = atlas._opponent(image)
    if code == "p05":
        groups = atlas._bins(rho, mask, 12)
        low = atlas._blur_masked(y, mask, max(0.1 * deq, 2.0))
        mid = atlas._blur_masked(y, mask, max(0.03 * deq, 1.5)) - low
        high = y - atlas._blur_masked(y, mask, max(0.02 * deq, 1.25))
        components = []
        for comp in (low, mid, high):
            target = np.sqrt(np.mean(comp[active] ** 2))
            out = comp.copy()
            for k in range(12):
                loc = (groups == k) & active
                if loc.any():
                    out[loc] *= target / max(
                        float(np.sqrt(np.mean(comp[loc] ** 2))), EPS
                    )
            components.append(out)
        result = _replace_luminance(image, sum(components))
        method = "equalize_low_mid_high_energy_across_12_radial_rings"
    elif code == "p10":
        inward = cv2.distanceTransform(active.astype(np.uint8), cv2.DIST_L2, 5)
        boundary = active & (inward <= 5)
        interior = active & (inward > 5)
        target = 0.5 * (float(np.mean(y[boundary])) + float(np.mean(y[interior])))
        new = y.copy()
        new[boundary] += target - np.mean(y[boundary])
        new[interior] += target - np.mean(y[interior])
        result = _replace_luminance(image, new)
        method = "equalize_boundary_and_interior_mean_luminance"
    elif code == "p15":
        radial = atlas._bins(rho, mask, 6)
        sector = _sector_map(active, 12)
        chroma = np.sqrt(a * a + b * b)
        scale = np.ones_like(y)
        for ring in range(6):
            locring = (radial == ring) & active
            target = float(np.median(chroma[locring])) if locring.any() else 0
            for s in range(12):
                loc = locring & (sector == s)
                if loc.any():
                    scale[loc] = target / max(float(np.median(chroma[loc])), EPS)
        mean = image.mean(2, keepdims=True)
        result = np.clip(mean + (image - mean) * scale[..., None], 0, 1).astype(
            np.float32
        )
        method = "equalize_angular_chroma_within_six_radial_rings"
    else:
        raise KeyError(code)
    result[~active] = image[~active]
    delta = result - image
    rms = float(np.sqrt(np.mean(delta[active] ** 2)))
    clip = float(np.mean((result[active] <= 0) | (result[active] >= 1)))
    return (
        result,
        {
            "method": method,
            "full_rms": rms,
            "clip_fraction": clip,
            "status": "pass" if np.isfinite(rms) else "failed",
            "reason": "",
        },
    )


def matched_control(image, full, mask, dose, seed):
    image = np.asarray(image, np.float32)
    active = np.asarray(mask) > 0.5
    physical = np.clip(image + dose * (full - image), 0, 1)
    target = float(np.sqrt(np.mean((physical[active] - image[active]) ** 2)))
    delta = full - image
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(128):
        dy = int(rng.integers(1, image.shape[0]))
        dx = int(rng.integers(1, image.shape[1]))
        candidate = np.roll(delta, (dy, dx), (0, 1))
        candidate[~active] = 0
        if np.sqrt(np.mean(candidate[active] ** 2)) <= EPS:
            continue
        ceiling = float(
            np.sqrt(
                np.mean(
                    (
                        np.clip(image + 65536 * dose * candidate, 0, 1)[active]
                        - image[active]
                    )
                    ** 2
                )
            )
        )
        if ceiling >= target:
            best = candidate
            break
    if best is None:
        best = rng.normal(size=image.shape).astype(np.float32) * active[..., None]
    (lo, hi) = (0.0, 1.0)
    while (
        hi < 65536
        and np.sqrt(
            np.mean(
                (np.clip(image + hi * dose * best, 0, 1)[active] - image[active]) ** 2
            )
        )
        < target
    ):
        hi *= 2
    for _ in range(30):
        mid = (lo + hi) / 2
        observed = float(
            np.sqrt(
                np.mean(
                    (np.clip(image + mid * dose * best, 0, 1)[active] - image[active])
                    ** 2
                )
            )
        )
        if observed < target:
            lo = mid
        else:
            hi = mid
    control = np.clip(image + (lo + hi) / 2 * dose * best, 0, 1)
    observed = float(np.sqrt(np.mean((control[active] - image[active]) ** 2)))
    err = abs(observed - target) / max(target, EPS)
    return (
        control.astype(np.float32),
        {
            "physical_rms": target,
            "control_rms": observed,
            "rms_relative_error": err,
            "status": "pass" if err <= 0.02 else "failed",
            "control_method": "spatial_displacement_energy_matched_adaptive",
        },
    )
