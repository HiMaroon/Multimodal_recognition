import csv
from pathlib import Path
import cv2
import numpy as np
from cellpose import models
from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops
from registration import read_image, write_image

CROP_SIZE = 128
HALF_CROP = CROP_SIZE // 2
MIN_AREA = 500
MAX_AREA = 12000
MAX_BBOX_SIZE = 120
MAX_FULL_EDGE_PIXELS = 29
MAX_CROP_EDGE_PIXELS = 4
MAX_NEIGHBOR_FRACTION = 0.2
MAX_ECCENTRICITY = 0.9
MIN_SOLIDITY = 0.9
MIN_EXTENT = 0.6
MIN_CIRCULARITY = 0.65
MAX_MATCH_DISTANCE = 25.0
MIN_MATCH_IOU = 0.05
MIN_MATCH_INTERSECTION = 100
T_CELLPOSE_DIAMETER = 60
T_CELLPOSE_FLOW_THRESHOLD = 0.6
T_CELLPOSE_CELLPROB_THRESHOLD = 0.5
RD_CELLPOSE_DIAMETER = 60
RD_CELLPOSE_FLOW_THRESHOLD = 0.6
RD_CELLPOSE_CELLPROB_THRESHOLD = 0.5


def warp_image(
    image: np.ndarray, shift_x: float, shift_y: float, nearest: bool = False
) -> np.ndarray:
    transform = np.array([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]], dtype=np.float32)
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.warpAffine(
        image,
        transform,
        (image.shape[1], image.shape[0]),
        flags=interpolation | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def warp_labels(labels: np.ndarray, shift_x: float, shift_y: float) -> np.ndarray:
    warped = warp_image(labels.astype(np.float32), shift_x, shift_y, nearest=True)
    return np.rint(warped).astype(np.uint16)


def label_properties(labels: np.ndarray) -> dict[int, dict[str, float]]:
    output = {}
    for prop in regionprops(labels):
        perimeter = max(float(prop.perimeter), 1e-06)
        output[int(prop.label)] = {
            "area": int(prop.area),
            "centroid_y": float(prop.centroid[0]),
            "centroid_x": float(prop.centroid[1]),
            "bbox_y1": int(prop.bbox[0]),
            "bbox_x1": int(prop.bbox[1]),
            "bbox_y2": int(prop.bbox[2]),
            "bbox_x2": int(prop.bbox[3]),
            "bbox_height": int(prop.bbox[2] - prop.bbox[0]),
            "bbox_width": int(prop.bbox[3] - prop.bbox[1]),
            "eccentricity": float(prop.eccentricity),
            "solidity": float(prop.solidity),
            "extent": float(prop.extent),
            "circularity": float(4 * np.pi * prop.area / (perimeter * perimeter)),
        }
    return output


def one_to_one_matches(
    rd_labels: np.ndarray,
    t_labels: np.ndarray,
    rd_props: dict[int, dict[str, float]],
    t_props: dict[int, dict[str, float]],
) -> dict[int, dict[str, float | int | bool]]:
    rd_ids = [
        label_id for (label_id, prop) in rd_props.items() if prop["area"] >= MIN_AREA
    ]
    t_ids = [
        label_id for (label_id, prop) in t_props.items() if prop["area"] >= MIN_AREA
    ]
    output = {
        label_id: {
            "t_label_id": 0,
            "center_distance_px": float("inf"),
            "iou": 0.0,
            "intersection_pixels": 0,
            "t_match_pass": False,
        }
        for label_id in rd_ids
    }
    if not rd_ids or not t_ids:
        return output
    rd_centers = np.array(
        [
            [rd_props[label]["centroid_y"], rd_props[label]["centroid_x"]]
            for label in rd_ids
        ]
    )
    t_centers = np.array(
        [
            [t_props[label]["centroid_y"], t_props[label]["centroid_x"]]
            for label in t_ids
        ]
    )
    distances = np.linalg.norm(rd_centers[:, None, :] - t_centers[None, :, :], axis=2)
    (rd_indices, t_indices) = linear_sum_assignment(distances)
    for rd_index, t_index in zip(rd_indices, t_indices):
        rd_id = rd_ids[int(rd_index)]
        t_id = t_ids[int(t_index)]
        distance = float(distances[rd_index, t_index])
        rd_component = rd_labels == rd_id
        t_component = t_labels == t_id
        intersection = int(np.count_nonzero(rd_component & t_component))
        union = int(np.count_nonzero(rd_component | t_component))
        iou = intersection / union if union else 0.0
        passed = bool(
            distance <= MAX_MATCH_DISTANCE
            and (iou >= MIN_MATCH_IOU or intersection >= MIN_MATCH_INTERSECTION)
        )
        output[rd_id] = {
            "t_label_id": t_id,
            "center_distance_px": distance,
            "iou": iou,
            "intersection_pixels": intersection,
            "t_match_pass": passed,
        }
    return output


def edge_counts(binary: np.ndarray) -> dict[str, int]:
    return {
        "top": int(np.count_nonzero(binary[0, :])),
        "bottom": int(np.count_nonzero(binary[-1, :])),
        "left": int(np.count_nonzero(binary[:, 0])),
        "right": int(np.count_nonzero(binary[:, -1])),
    }


def crop_bounds(
    center_y: int, center_x: int, height: int, width: int
) -> tuple[int, int, int, int] | None:
    (y1, y2) = (center_y - HALF_CROP, center_y + HALF_CROP)
    (x1, x2) = (center_x - HALF_CROP, center_x + HALF_CROP)
    if y1 < 0 or x1 < 0 or y2 > height or (x2 > width):
        return None
    return (y1, y2, x1, x2)


def image_evidence(image: np.ndarray, component: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inner = cv2.erode(component.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    ring = (
        cv2.dilate(component.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    ) & ~component
    if np.count_nonzero(inner) < 20 or np.count_nonzero(ring) < 20:
        return {"delta": 0.0, "deviation_ratio": 0.0, "texture_ratio": 0.0}
    inside_values = gray[inner].astype(np.float32)
    ring_values = gray[ring].astype(np.float32)
    ring_median = float(np.median(ring_values))
    delta = float(abs(inside_values.mean() - ring_values.mean()))
    deviation_ratio = float(
        np.mean(np.abs(inside_values - ring_median))
        / (np.mean(np.abs(ring_values - ring_median)) + 1e-06)
    )
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    texture_ratio = float(laplacian[inner].mean() / (laplacian[ring].mean() + 1e-06))
    return {
        "delta": delta,
        "deviation_ratio": deviation_ratio,
        "texture_ratio": texture_ratio,
    }


def relative_paths(
    split: str, cell_type: str, label_id: int, group_index: int
) -> dict[str, Path]:
    root = Path(split) / cell_type
    return {
        "RD": root / "RD" / f"{label_id}_G-{cell_type}-RD-{group_index}.jpg",
        "RB": root / "RB" / f"{label_id}_G-{cell_type}-RB-{group_index}.jpg",
        "T": root / "T" / f"{label_id}_G-{cell_type}-T-{group_index}.jpg",
        "mask": root / "mask" / f"{label_id}_G-{cell_type}-RD-{group_index}_mask.bmp",
    }


def save_sample(
    root: Path,
    paths: dict[str, Path],
    crops: dict[str, np.ndarray],
    mask_crop: np.ndarray,
    masked: bool,
) -> None:
    for modality in ("RD", "RB", "T"):
        image = crops[modality]
        if masked:
            image = (image * mask_crop[:, :, None]).astype(np.uint8)
        write_image(root / paths[modality], image)
    write_image(root / paths["mask"], (mask_crop * 255).astype(np.uint8))


def write_group_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def process_group(
    audit: dict[str, str],
    model: models.Cellpose,
    background_root: Path,
    masked_root: Path,
    quarantine_root: Path,
) -> list[dict[str, object]]:
    batch_name = audit["batch_folder"]
    group_index = int(audit["group_index"])
    split = audit["split"]
    cell_type = audit["cell_type"]
    (rb_shift_x, rb_shift_y) = (float(audit["rb_shift_x"]), float(audit["rb_shift_y"]))
    (t_shift_x, t_shift_y) = (float(audit["t_shift_x"]), float(audit["t_shift_y"]))
    images_native = {
        "RD": read_image(Path(audit["raw_rd"])),
        "RB": read_image(Path(audit["raw_rb"])),
        "T": read_image(Path(audit["raw_t"])),
    }
    images = {
        "RD": images_native["RD"],
        "RB": warp_image(images_native["RB"], rb_shift_x, rb_shift_y),
        "T": warp_image(images_native["T"], t_shift_x, t_shift_y),
    }
    (height, width) = images["RD"].shape[:2]
    (rd_labels, _, _, _) = model.eval(
        images_native["RD"],
        diameter=RD_CELLPOSE_DIAMETER,
        flow_threshold=RD_CELLPOSE_FLOW_THRESHOLD,
        invert=False,
        cellprob_threshold=RD_CELLPOSE_CELLPROB_THRESHOLD,
        channels=[0, 0],
    )
    (t_labels_native, _, _, _) = model.eval(
        images_native["T"],
        diameter=T_CELLPOSE_DIAMETER,
        flow_threshold=T_CELLPOSE_FLOW_THRESHOLD,
        invert=True,
        cellprob_threshold=T_CELLPOSE_CELLPROB_THRESHOLD,
        channels=[0, 0],
    )
    t_labels = warp_labels(t_labels_native, t_shift_x, t_shift_y)
    rd_props = label_properties(rd_labels)
    t_props = label_properties(t_labels)
    matches = one_to_one_matches(rd_labels, t_labels, rd_props, t_props)
    mask_group_root = background_root.parent / "instance_labels" / batch_name
    write_image(
        mask_group_root / f"G-{cell_type}-RD-{group_index}_labels.png",
        rd_labels.astype(np.uint16),
    )
    write_image(
        mask_group_root / f"G-{cell_type}-T-{group_index}_labels_native.png",
        t_labels_native.astype(np.uint16),
    )
    write_image(
        mask_group_root / f"G-{cell_type}-T-{group_index}_labels_aligned.png", t_labels
    )
    rows = []
    for rd_id, prop in sorted(rd_props.items()):
        component = rd_labels == rd_id
        full_edges = edge_counts(component)
        center_y = int(round((prop["bbox_y1"] + prop["bbox_y2"] - 1) / 2))
        center_x = int(round((prop["bbox_x1"] + prop["bbox_x2"] - 1) / 2))
        bounds = crop_bounds(center_y, center_x, height, width)
        match = matches.get(
            rd_id,
            {
                "t_label_id": 0,
                "center_distance_px": float("inf"),
                "iou": 0.0,
                "intersection_pixels": 0,
                "t_match_pass": False,
            },
        )
        reasons = []
        if prop["area"] < MIN_AREA:
            reasons.append("area_lt_500")
        if prop["area"] > MAX_AREA:
            reasons.append("area_gt_12000")
        if max(full_edges.values()) > MAX_FULL_EDGE_PIXELS:
            reasons.append("full_image_edge")
        if prop["bbox_height"] > MAX_BBOX_SIZE or prop["bbox_width"] > MAX_BBOX_SIZE:
            reasons.append("bbox_gt_120")
        if prop["eccentricity"] > MAX_ECCENTRICITY:
            reasons.append("high_eccentricity")
        if prop["solidity"] < MIN_SOLIDITY:
            reasons.append("low_solidity")
        if prop["extent"] < MIN_EXTENT:
            reasons.append("low_extent")
        if prop["circularity"] < MIN_CIRCULARITY:
            reasons.append("low_circularity")
        if not match["t_match_pass"]:
            reasons.append("no_t_consensus")
        if bounds is None:
            reasons.append("crop_out_of_bounds")
        crop_edges = {"top": 0, "bottom": 0, "left": 0, "right": 0}
        neighbor_pixels = 0
        neighbor_fraction = 0.0
        rd_evidence = {"delta": 0.0, "deviation_ratio": 0.0, "texture_ratio": 0.0}
        rb_evidence = rd_evidence.copy()
        t_evidence = rd_evidence.copy()
        crops = None
        mask_crop = None
        if bounds is not None:
            (y1, y2, x1, x2) = bounds
            mask_crop = component[y1:y2, x1:x2].astype(np.uint8)
            crop_edges = edge_counts(mask_crop)
            if max(crop_edges.values()) > MAX_CROP_EDGE_PIXELS:
                reasons.append("crop_mask_edge")
            label_crop = rd_labels[y1:y2, x1:x2]
            neighbor_pixels = int(
                np.count_nonzero((label_crop > 0) & (label_crop != rd_id))
            )
            neighbor_fraction = neighbor_pixels / max(float(prop["area"]), 1.0)
            if neighbor_fraction > MAX_NEIGHBOR_FRACTION:
                reasons.append("neighbor_contamination")
            crops = {
                modality: image[y1:y2, x1:x2] for (modality, image) in images.items()
            }
            local_component = mask_crop.astype(bool)
            rd_evidence = image_evidence(crops["RD"], local_component)
            rb_evidence = image_evidence(crops["RB"], local_component)
            t_evidence = image_evidence(crops["T"], local_component)
            rd_signal_pass = bool(
                rd_evidence["delta"] >= 8
                or rd_evidence["deviation_ratio"] >= 1.3
                or rd_evidence["texture_ratio"] >= 1.15
            )
            t_signal_pass = bool(
                t_evidence["delta"] >= 5
                or t_evidence["deviation_ratio"] >= 1.3
                or t_evidence["texture_ratio"] >= 1.15
            )
            if not rd_signal_pass:
                reasons.append("weak_rd_signal")
            if not t_signal_pass:
                reasons.append("weak_t_signal")
        status = "kept" if not reasons else "review"
        paths = relative_paths(split, cell_type, rd_id, group_index)
        if crops is not None and mask_crop is not None:
            if status == "kept":
                save_sample(background_root, paths, crops, mask_crop, masked=False)
                save_sample(masked_root, paths, crops, mask_crop, masked=True)
            elif prop["area"] >= MIN_AREA:
                save_sample(quarantine_root, paths, crops, mask_crop, masked=False)
        rows.append(
            {
                "batch_folder": batch_name,
                "group_index": group_index,
                "split": split,
                "cell_type": cell_type,
                "rd_label_id": rd_id,
                "status": status,
                "review_reasons": "|".join(reasons),
                **{f"rd_{name}": value for (name, value) in prop.items()},
                "t_label_id": match["t_label_id"],
                "center_distance_px": match["center_distance_px"],
                "iou": match["iou"],
                "intersection_pixels": match["intersection_pixels"],
                "t_match_pass": match["t_match_pass"],
                "full_top_pixels": full_edges["top"],
                "full_bottom_pixels": full_edges["bottom"],
                "full_left_pixels": full_edges["left"],
                "full_right_pixels": full_edges["right"],
                "crop_top_pixels": crop_edges["top"],
                "crop_bottom_pixels": crop_edges["bottom"],
                "crop_left_pixels": crop_edges["left"],
                "crop_right_pixels": crop_edges["right"],
                "neighbor_pixels": neighbor_pixels,
                "neighbor_fraction": neighbor_fraction,
                "rd_delta": rd_evidence["delta"],
                "rd_deviation_ratio": rd_evidence["deviation_ratio"],
                "rd_texture_ratio": rd_evidence["texture_ratio"],
                "rb_delta": rb_evidence["delta"],
                "rb_deviation_ratio": rb_evidence["deviation_ratio"],
                "rb_texture_ratio": rb_evidence["texture_ratio"],
                "t_delta": t_evidence["delta"],
                "t_deviation_ratio": t_evidence["deviation_ratio"],
                "t_texture_ratio": t_evidence["texture_ratio"],
                "rb_shift_x": rb_shift_x,
                "rb_shift_y": rb_shift_y,
                "rb_registration_ecc": float(audit["rb_ecc"]),
                "rb_registration_low_confidence": float(audit["rb_ecc"]) < 0.7,
                "t_shift_x": t_shift_x,
                "t_shift_y": t_shift_y,
                "t_registration_ecc": float(audit["t_ecc"]),
                "rd_path": str(paths["RD"]),
                "rb_path": str(paths["RB"]),
                "t_path": str(paths["T"]),
                "mask_path": str(paths["mask"]),
            }
        )
    return rows
