import cv2
import numpy as np
import intensity_kernels as atlas


def control_layout(
    image: np.ndarray, full: np.ndarray, mask: np.ndarray, seed: int, code: str
):
    active = mask > 0.5
    rng = np.random.default_rng(seed)
    delta = full - image
    control_delta = np.zeros_like(delta)
    (_, _, _, deq) = atlas._geometry(mask.astype(np.float32))
    if code in {"p08"}:
        groups = atlas._depth_bins_with_guard(mask.astype(np.float32), deq, 12)
        shift = int(rng.integers(1, 12))
        for layer in range(12):
            target = (groups == layer) & active
            source = (groups == (layer + shift) % 12) & active
            if target.any() and source.any():
                control_delta[target] = np.median(delta[source], axis=0)
        method = "nonzero_cyclic_depth_offset_permutation"
    else:
        inward = cv2.distanceTransform(active.astype(np.uint8), cv2.DIST_L2, 5)
        width = max(0.1 * deq, 2.0)
        interface = active & (inward <= width)
        (yy, xx) = np.indices(active.shape, dtype=np.float32)
        (cy, cx) = np.mean(np.where(active), axis=1)
        sectors = np.minimum(
            ((np.arctan2(yy - cy, xx - cx) + np.pi) / (2 * np.pi) * 12).astype(int), 11
        )
        shift = int(rng.integers(1, 12))
        for sector in range(12):
            target = interface & (sectors == sector)
            source = interface & (sectors == (sector + shift) % 12)
            if target.any() and source.any():
                control_delta[target] = np.median(delta[source], axis=0)
        method = "nonzero_cyclic_interface_angular_offset"
    return (control_delta, method)


def adaptive_control(
    image: np.ndarray,
    full: np.ndarray,
    mask: np.ndarray,
    dose: float,
    seed: int,
    code: str,
):
    active = mask > 0.5
    (delta, method) = control_layout(image, full, mask, seed, code)
    original_active = image[active]
    physical_active = np.clip(original_active + dose * (full - image)[active], 0, 1)
    control_delta_active = delta[active]
    target = float(np.sqrt(np.mean((physical_active - original_active) ** 2)))
    if target <= atlas.EPS:
        return (
            image.copy(),
            {
                "physical_rms": target,
                "control_rms": 0.0,
                "rms_relative_error": 0.0,
                "solver_max_observed": 0.0,
                "control_method": method,
                "status": "pass",
            },
        )
    (lo, hi) = (0.0, 1.0)
    observed = 0.0
    while hi < 65536:
        candidate_active = np.clip(
            original_active + hi * dose * control_delta_active, 0, 1
        )
        observed = float(np.sqrt(np.mean((candidate_active - original_active) ** 2)))
        if observed >= target:
            break
        hi *= 2.0
    maximum = observed
    chosen_scale = None
    for _ in range(18):
        scale = (lo + hi) / 2
        candidate_active = np.clip(
            original_active + scale * dose * control_delta_active, 0, 1
        )
        observed = float(np.sqrt(np.mean((candidate_active - original_active) ** 2)))
        if abs(observed - target) / max(target, atlas.EPS) <= 0.002:
            chosen_scale = scale
            break
        if observed < target:
            lo = scale
        else:
            hi = scale
    if chosen_scale is None:
        chosen_scale = (lo + hi) / 2
    control = image.copy()
    control[active] = np.clip(
        original_active + chosen_scale * dose * control_delta_active, 0, 1
    )
    control = control.astype(np.float32)
    observed = float(np.sqrt(np.mean((control[active] - original_active) ** 2)))
    error = abs(observed - target) / max(target, atlas.EPS)
    return (
        control,
        {
            "physical_rms": target,
            "control_rms": observed,
            "rms_relative_error": error,
            "solver_max_observed": maximum,
            "control_method": method,
            "status": "pass" if error <= 0.02 else "failed",
        },
    )
