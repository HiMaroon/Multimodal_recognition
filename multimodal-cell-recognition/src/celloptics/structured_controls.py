import numpy as np
import intensity_kernels as atlas


def p01_adaptive_control(
    image: np.ndarray, full: np.ndarray, mask: np.ndarray, dose: float, seed: int
):
    active = mask > 0.5
    rng = np.random.default_rng(seed)
    delta = full - image
    noise = rng.normal(size=mask.shape).astype(np.float32)
    deq = atlas._geometry(mask.astype(np.float32))[3]
    gaussian = atlas._blur_masked(noise, mask.astype(np.float32), max(0.08 * deq, 1.0))
    gaussian[~active] = 0
    gaussian[active] -= gaussian[active].mean()
    template = delta[active].mean(axis=0)
    direction = template / max(np.sqrt(np.mean(template**2)), atlas.EPS)
    physical = np.clip(image + dose * delta, 0, 1)
    target = float(np.sqrt(np.mean((physical[active] - image[active]) ** 2)))
    z = gaussian.copy()
    z[active] /= max(float(np.std(z[active])), atlas.EPS)
    fields = [("gaussian", gaussian)]
    for strength in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0):
        exp_field = np.zeros_like(z)
        clipped_z = np.clip(strength * z[active], -12, 12)
        exp_field[active] = np.exp(clipped_z)
        exp_field[active] -= exp_field[active].mean()
        fields.extend(
            (
                (f"exp_{strength:g}_positive", exp_field),
                (f"exp_{strength:g}_negative", -exp_field),
            )
        )
    candidates = []
    target_raw_rms = np.sqrt(np.mean(delta[active] ** 2))
    for field_name, field in fields:
        raw_scale = target_raw_rms / max(
            np.sqrt(np.mean(field[active] ** 2)), atlas.EPS
        )
        proposed_delta = field[..., None] * raw_scale * direction
        saturated = np.clip(image + 65536 * dose * proposed_delta, 0, 1)
        maximum = float(np.sqrt(np.mean((saturated[active] - image[active]) ** 2)))
        candidates.append((maximum, field_name, proposed_delta))
    viable = [item for item in candidates if item[0] >= target]
    (solver_max_observed, field_name, control_delta) = (
        viable[0] if viable else max(candidates, key=lambda item: item[0])
    )
    (lo, hi) = (0.0, 1.0)
    observed = 0.0
    while hi < 65536:
        candidate = np.clip(image + hi * dose * control_delta, 0, 1)
        observed = float(np.sqrt(np.mean((candidate[active] - image[active]) ** 2)))
        if observed >= target:
            break
        hi *= 2
    for _ in range(32):
        scale = (lo + hi) / 2
        candidate = np.clip(image + scale * dose * control_delta, 0, 1)
        observed = float(np.sqrt(np.mean((candidate[active] - image[active]) ** 2)))
        if observed < target:
            lo = scale
        else:
            hi = scale
    control = np.clip(image + (lo + hi) / 2 * dose * control_delta, 0, 1)
    observed = float(np.sqrt(np.mean((control[active] - image[active]) ** 2)))
    error = abs(observed - target) / max(target, atlas.EPS)
    return (
        control.astype(np.float32),
        {
            "physical_rms": target,
            "control_rms": observed,
            "rms_relative_error": error,
            "control_method": f"smooth_zero_mean_global_strength_control_{field_name}_adaptive_solver",
            "solver_max_observed": solver_max_observed,
            "status": "pass" if error <= 0.02 else "failed",
        },
    )


def structured_adaptive_control(
    image: np.ndarray,
    full: np.ndarray,
    mask: np.ndarray,
    dose: float,
    seed: int,
    code: str,
):
    (image, full) = (image.astype(np.float32), full.astype(np.float32))
    active = mask > 0.5
    rng = np.random.default_rng(seed)
    delta = full - image
    control_delta = np.zeros_like(delta)
    (rho, _, _, deq) = atlas._geometry(mask.astype(np.float32))
    if code in {"p04", "p06"}:
        n = 12 if code == "p04" else 6
        groups = atlas._bins(rho, mask, n)
        shift = int(rng.integers(1, n))
        for k in range(n):
            (target_group, source_group) = (groups == k, groups == (k + shift) % n)
            if target_group.any() and source_group.any():
                control_delta[target_group] = delta[source_group].mean(axis=0)
        method = "nonzero_cyclic_radial_offset_permutation"
    elif code in {"p07", "p09"}:
        n = 12
        groups = atlas._depth_bins_with_guard(mask.astype(np.float32), deq, n)
        shift = int(rng.integers(1, n))
        for k in range(n):
            (target_group, source_group) = (groups == k, groups == (k + shift) % n)
            if target_group.any() and source_group.any():
                control_delta[target_group] = delta[source_group].mean(axis=0)
        method = "nonzero_cyclic_depth_offset_permutation"
    elif code in {"p13"}:
        groups = atlas._bins(rho, mask, 12)
        for k in range(12):
            locations = np.flatnonzero((groups == k) & active)
            if len(locations) > 1:
                control_delta.reshape(-1, 3)[locations] = delta.reshape(-1, 3)[
                    np.roll(locations, int(rng.integers(1, len(locations))))
                ]
        method = "within_radial_stratum_circular_shift"
    else:
        dy = int(rng.integers(1, max(2, image.shape[0])))
        dx = int(rng.integers(1, max(2, image.shape[1])))
        control_delta = np.roll(delta, shift=(dy, dx), axis=(0, 1)) * mask[..., None]
        method = "spatial_displacement_energy_matched"
    physical = np.clip(image + dose * delta, 0, 1)
    target = float(np.sqrt(np.mean((physical[active] - image[active]) ** 2)))
    (lo, hi) = (0.0, 1.0)
    observed = 0.0
    while hi < 65536:
        candidate = np.clip(image + hi * dose * control_delta, 0, 1)
        observed = float(np.sqrt(np.mean((candidate[active] - image[active]) ** 2)))
        if observed >= target:
            break
        hi *= 2
    solver_max_observed = observed
    for _ in range(32):
        scale = (lo + hi) / 2
        candidate = np.clip(image + scale * dose * control_delta, 0, 1)
        observed = float(np.sqrt(np.mean((candidate[active] - image[active]) ** 2)))
        if observed < target:
            lo = scale
        else:
            hi = scale
    control = np.clip(image + (lo + hi) / 2 * dose * control_delta, 0, 1)
    observed = float(np.sqrt(np.mean((control[active] - image[active]) ** 2)))
    error = abs(observed - target) / max(target, atlas.EPS)
    return (
        control.astype(np.float32),
        {
            "physical_rms": target,
            "control_rms": observed,
            "rms_relative_error": error,
            "control_method": f"{method}_adaptive_solver",
            "solver_max_observed": solver_max_observed,
            "status": "pass" if error <= 0.02 else "failed",
        },
    )
