from pathlib import Path
import cv2
import numpy as np


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (ok, encoded) = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(path)


def registration_feature(
    image: np.ndarray, scale: float = 0.5, kind: str = "std"
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    reduced = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    reduced = cv2.GaussianBlur(reduced, (0, 0), 1.2)
    values = reduced.astype(np.float32)
    if kind == "std":
        local_mean = cv2.GaussianBlur(values, (0, 0), 5)
        local_square_mean = cv2.GaussianBlur(values * values, (0, 0), 5)
        feature = np.sqrt(np.maximum(local_square_mean - local_mean * local_mean, 0))
    elif kind == "gradient":
        grad_x = cv2.Sobel(values, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(values, cv2.CV_32F, 0, 1, ksize=3)
        feature = cv2.magnitude(grad_x, grad_y)
    elif kind == "dog":
        feature = np.abs(values - cv2.GaussianBlur(values, (0, 0), 12))
    else:
        raise ValueError(f"Unknown registration feature: {kind}")
    upper = float(np.percentile(feature, 99))
    feature = np.clip(feature, 0, upper) / (upper + 1e-06)
    feature = cv2.GaussianBlur(feature, (0, 0), 2)
    (height, width) = feature.shape
    border_x = max(10, int(round(width * 0.025)))
    border_top = max(10, int(round(height * 0.04)))
    border_bottom = max(30, int(round(height * 0.125)))
    feature[:border_top, :] = 0
    feature[height - border_bottom :, :] = 0
    feature[:, :border_x] = 0
    feature[:, width - border_x :] = 0
    return feature.astype(np.float32)


def estimate_translation(
    fixed_image: np.ndarray, moving_image: np.ndarray, scale: float = 0.5
) -> dict[str, float | str | bool]:
    if fixed_image.shape != moving_image.shape:
        raise ValueError(
            f"Registration image shapes differ: {fixed_image.shape} vs {moving_image.shape}"
        )
    candidates = []
    for kind in ("std", "gradient", "dog"):
        fixed = registration_feature(fixed_image, scale=scale, kind=kind)
        moving = registration_feature(moving_image, scale=scale, kind=kind)
        window = cv2.createHanningWindow((fixed.shape[1], fixed.shape[0]), cv2.CV_32F)
        (phase_shift, phase_response) = cv2.phaseCorrelate(fixed, moving, window)
        warp = np.array(
            [[1.0, 0.0, phase_shift[0]], [0.0, 1.0, phase_shift[1]]], dtype=np.float32
        )
        try:
            (ecc, refined) = cv2.findTransformECC(
                fixed,
                moving,
                warp,
                cv2.MOTION_TRANSLATION,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-06),
                None,
                5,
            )
            shift_x = float(refined[0, 2] / scale)
            shift_y = float(refined[1, 2] / scale)
            candidates.append(
                {
                    "method": kind,
                    "shift_x": shift_x,
                    "shift_y": shift_y,
                    "ecc": float(ecc),
                    "phase_response": float(phase_response),
                }
            )
        except cv2.error:
            continue
    if not candidates:
        return {
            "method": "failed",
            "shift_x": 0.0,
            "shift_y": 0.0,
            "ecc": 0.0,
            "phase_response": 0.0,
            "method_consensus_px": float("inf"),
            "passed": False,
        }
    plausible = [
        row
        for row in candidates
        if abs(row["shift_x"]) <= 300 and abs(row["shift_y"]) <= 300
    ]
    if not plausible:
        plausible = candidates
    best = max(plausible, key=lambda row: row["ecc"])
    distances = [
        float(
            np.hypot(row["shift_x"] - best["shift_x"], row["shift_y"] - best["shift_y"])
        )
        for row in plausible
        if row is not best
    ]
    consensus = float(np.median(distances)) if distances else 0.0
    passed = bool(
        best["ecc"] >= 0.7
        and abs(best["shift_x"]) <= 300
        and (abs(best["shift_y"]) <= 300)
        and (consensus <= 25)
    )
    return {**best, "method_consensus_px": consensus, "passed": passed}
