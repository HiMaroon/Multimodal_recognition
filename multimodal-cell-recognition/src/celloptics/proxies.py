import json
import hashlib
import cv_config as cfg
import intensity_descriptors
import intensity_kernels
import spatial_descriptors
import spatial_interventions
import boundary_kernels
import structured_controls
import boundary_controls

REGISTRY = json.loads((cfg.RELEASE_ROOT / "configs/proxies.json").read_text())
BY_ID = {r["id"]: r for r in REGISTRY}
SPATIAL = {"p05", "p10", "p15"}
BOUNDARY = {"p08", "p12"}


def descriptor(image, mask, code):
    if code not in BY_ID:
        raise KeyError(code)
    if code in SPATIAL:
        return spatial_descriptors.extract_single(image, mask, code)
    if code in BOUNDARY:
        return boundary_kernels.extract_descriptor(image, mask, code)
    return intensity_descriptors.extract_descriptor(image, mask, code)


def neutralize(image, mask, code, seed=0):
    if code not in BY_ID:
        raise KeyError(code)
    if code in SPATIAL:
        return spatial_interventions.neutralise_full(image, mask, code)
    kernel = boundary_kernels if code in BOUNDARY else intensity_kernels
    return kernel.neutralise_full(image, mask, code, seed=seed)


def control(image, full, mask, dose, seed, code):
    max_retries = 64 if code in BOUNDARY else 256
    for retry in range(max_retries):
        stride = 100000003 if code in BOUNDARY else 1000003
        s = seed + retry * stride
        if code in SPATIAL:
            (altered, qc) = spatial_interventions.matched_control(
                image, full, mask, dose, s
            )
        elif code in BOUNDARY:
            (altered, qc) = boundary_controls.adaptive_control(
                image, full, mask, dose, s, code
            )
        elif code == "p01":
            (altered, qc) = structured_controls.p01_adaptive_control(
                image, full, mask, dose, s
            )
        else:
            (altered, qc) = structured_controls.structured_adaptive_control(
                image, full, mask, dose, s, code
            )
        if qc["status"] == "pass":
            break
    qc["control_retry_count"] = retry
    return (altered, qc)


def control_seed(code, sample_id, dose, replicate):
    offset = BY_ID[code]["control_seed_offset"]
    key = f"{cfg.PROTOCOL['proxy_cnn_seed']}|{offset}|{code}|{sample_id}|{dose:.2f}|{replicate}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
