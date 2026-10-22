import csv
import gc
import json
import platform
import time
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
import cv_config as cfg
from cv_data import BatchSpec, ManifestDataset, rows_for_fold
from reproducibility import configure, seed_worker
from protocol_guard import validate_training_rows, sha256


def log(message: str) -> None:
    print(message, flush=True)


def set_seed(seed: int) -> None:
    configure(seed)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def make_autocast(device: torch.device):
    if not (cfg.USE_AMP and device.type == "cuda"):
        return nullcontext()
    try:
        return torch.amp.autocast(device_type="cuda", enabled=True)
    except Exception:
        return torch.cuda.amp.autocast(enabled=True)


def make_grad_scaler(device: torch.device):
    enabled = bool(cfg.USE_AMP and device.type == "cuda")
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0, 1.0, row_sums)
    return probs / row_sums


def safe_macro_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    try:
        return float(
            roc_auc_score(
                labels, normalize_probs(probs), average="macro", multi_class="ovr"
            )
        )
    except ValueError:
        return float("nan")


def resolve_num_workers(split_name: str) -> int:
    requested = int(cfg.NUM_WORKERS)
    if not platform.system().lower().startswith("win"):
        return max(0, requested)
    if split_name == "train":
        return max(0, min(requested, cfg.WINDOWS_TRAIN_WORKERS))
    return max(0, min(requested, cfg.WINDOWS_EVAL_WORKERS))


def build_loader(
    rows: list[dict],
    spec: BatchSpec,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    split_name: str,
) -> DataLoader:
    kwargs = {
        "dataset": ManifestDataset(rows, spec),
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": resolve_num_workers(split_name),
        "pin_memory": bool(cfg.USE_PIN_MEMORY and device.type == "cuda"),
        "drop_last": bool(split_name == "train" and len(rows) % batch_size == 1),
        "worker_init_fn": seed_worker,
        "generator": torch.Generator().manual_seed(torch.initial_seed()),
    }
    if kwargs["num_workers"] > 0 and (not platform.system().lower().startswith("win")):
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def release_loader(loader: DataLoader) -> None:
    try:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None and hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()
    except Exception:
        pass
    del loader
    gc.collect()


def run_epoch(model, optimizer, loader, device, loss_cls_func, epoch: int) -> dict:
    model.train()
    scaler = getattr(run_epoch, "_scaler", None)
    if scaler is None:
        scaler = make_grad_scaler(device)
        run_epoch._scaler = scaler
    totals = {"loss": 0.0, "cls_loss": 0.0}
    (probs_all, labels_all) = ([], [])
    n_seen = 0
    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["images"].float().to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with make_autocast(device):
            outputs = model(images)
            loss = loss_cls_func(outputs["cls_pred"], labels)
            cls_loss = loss.detach()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        bs = images.size(0)
        n_seen += bs
        totals["loss"] += loss.item() * bs
        totals["cls_loss"] += cls_loss.item() * bs
        probs_all.append(
            torch.softmax(outputs["cls_pred"].detach().float(), dim=1).cpu().numpy()
        )
        labels_all.append(labels.detach().cpu().numpy())
        if cfg.PRINT_EVERY_N_BATCHES and (
            batch_idx == 1
            or batch_idx % cfg.PRINT_EVERY_N_BATCHES == 0
            or batch_idx == len(loader)
        ):
            log(
                f"[Epoch {epoch:03d}] train batch {batch_idx:04d}/{len(loader):04d} loss={totals['loss'] / max(n_seen, 1):.6f}"
            )
    probs = normalize_probs(np.concatenate(probs_all, axis=0))
    labels_np = np.concatenate(labels_all, axis=0)
    preds = np.argmax(probs, axis=1)
    n = n_seen
    return {
        "loss": totals["loss"] / n,
        "cls_loss": totals["cls_loss"] / n,
        "acc": float(np.mean(preds == labels_np)),
        "auc_macro_ovr": safe_macro_auc(labels_np, probs),
    }


def evaluate_split(model, loader, device, loss_cls_func, split_name: str) -> dict:
    model.eval()
    totals = {"loss": 0.0, "cls_loss": 0.0}
    (probs_all, labels_all, names_all, sample_ids) = ([], [], [], [])
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            images = batch["images"].float().to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with make_autocast(device):
                outputs = model(images)
                loss = loss_cls_func(outputs["cls_pred"], labels)
                cls_loss = loss.detach()
            bs = images.size(0)
            totals["loss"] += loss.item() * bs
            totals["cls_loss"] += cls_loss.item() * bs
            probs_all.append(
                torch.softmax(outputs["cls_pred"].float(), dim=1).cpu().numpy()
            )
            labels_all.append(labels.cpu().numpy())
            names_all.extend(batch["image_name"])
            sample_ids.extend(batch["sample_id"])
            if cfg.PRINT_EVERY_N_BATCHES and (
                batch_idx == 1
                or batch_idx % cfg.PRINT_EVERY_N_BATCHES == 0
                or batch_idx == len(loader)
            ):
                log(f"[{split_name}] eval batch {batch_idx:04d}/{len(loader):04d}")
    probs = normalize_probs(np.concatenate(probs_all, axis=0))
    labels_np = np.concatenate(labels_all, axis=0)
    preds = np.argmax(probs, axis=1)
    n = len(loader.dataset)
    overall = {
        "loss": float(totals["loss"] / n),
        "cls_loss": float(totals["cls_loss"] / n),
        "accuracy": float(accuracy_score(labels_np, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels_np, preds)),
        "macro_precision": float(
            precision_score(labels_np, preds, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(labels_np, preds, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(labels_np, preds, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(labels_np, preds, average="weighted", zero_division=0)
        ),
        "macro_auc_ovr": safe_macro_auc(labels_np, probs),
    }
    try:
        overall["micro_auc_ovr"] = float(
            roc_auc_score(labels_np, probs, average="micro", multi_class="ovr")
        )
    except ValueError:
        overall["micro_auc_ovr"] = float("nan")
    (pr, rc, f1, support) = precision_recall_fscore_support(
        labels_np, preds, labels=[0, 1, 2], zero_division=0
    )
    per_class = []
    for idx, class_name in enumerate(cfg.CLASS_ORDER):
        per_class.append(
            {
                "class_name": class_name,
                "precision": float(pr[idx]),
                "recall": float(rc[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
        )
    return {
        "overall": overall,
        "per_class_multiclass": per_class,
        "confusion_matrix": confusion_matrix(
            labels_np, preds, labels=[0, 1, 2]
        ).tolist(),
        "labels": labels_np.tolist(),
        "preds": preds.tolist(),
        "probs": probs.tolist(),
        "image_names": names_all,
        "sample_ids": sample_ids,
    }


def save_metrics(split_name: str, metrics: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"metrics_{split_name}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out_dir / f"overall_metrics_{split_name}.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics["overall"].items():
            writer.writerow([key, value])
    with (out_dir / f"predictions_{split_name}.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample_id",
                "image_name",
                "label",
                "pred",
                "prob_MCF7",
                "prob_10A",
                "prob_HCT",
            ]
        )
        for sample_id, name, label, pred, prob in zip(
            metrics["sample_ids"],
            metrics["image_names"],
            metrics["labels"],
            metrics["preds"],
            metrics["probs"],
        ):
            writer.writerow([sample_id, name, label, pred] + list(prob))


def save_history(history: list[dict], out_dir: Path) -> None:
    with (out_dir / "training_history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "val_loss",
                "val_macro_f1",
                "val_macro_auc_ovr",
                "seconds",
                "lr",
            ]
        )
        for row in history:
            writer.writerow(
                [
                    row["epoch"],
                    row["train"]["loss"],
                    row["val"]["loss"],
                    row["val"]["macro_f1"],
                    row["val"]["macro_auc_ovr"],
                    row["seconds"],
                    row["lr"],
                ]
            )


def train_fold(
    run_name: str,
    fold: int,
    modalities: tuple[str, ...],
    model_factory,
    out_dir: Path,
    max_epochs: int | None = None,
    min_epoch_for_best: int | None = None,
) -> dict:
    set_seed(cfg.SEED + fold)
    if hasattr(run_epoch, "_scaler"):
        delattr(run_epoch, "_scaler")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_cls_func = nn.CrossEntropyLoss()
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "experiment_result.json"
    train_rows = rows_for_fold(fold, train=True)
    val_rows = rows_for_fold(fold, train=False)
    validate_training_rows(train_rows, val_rows)
    if result_path.exists():
        qa_path = out_dir / "checkpoint_reload_qa.json"
        if not qa_path.exists():
            raise RuntimeError(
                f"Existing result has no checkpoint reload QA: {result_path}"
            )
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        if qa.get("status") != "pass":
            raise RuntimeError(f"Existing result failed checkpoint QA: {qa_path}")
        if qa.get("checkpoint_sha256") != sha256(out_dir / "best_model_state.pt"):
            raise RuntimeError(f"Checkpoint changed after QA: {out_dir}")
        log(f"[Skip Existing] {result_path}")
        return json.loads(result_path.read_text(encoding="utf-8"))
    train_loader = build_loader(
        train_rows,
        BatchSpec(modalities=modalities, train=True),
        cfg.FIXED_BATCH_SIZE,
        True,
        device,
        "train",
    )
    val_loader = build_loader(
        val_rows,
        BatchSpec(modalities=modalities, train=False),
        cfg.FIXED_BATCH_SIZE,
        False,
        device,
        "val",
    )
    model = model_factory().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=cfg.SCHEDULER_PATIENCE,
        factor=cfg.SCHEDULER_FACTOR,
        min_lr=cfg.SCHEDULER_MIN_LR,
    )
    history = []
    best_score = -1.0
    best_epoch = 0
    best_state = None
    start = time.time()
    epochs = int(max_epochs if max_epochs is not None else cfg.EPOCHS)
    min_best = int(
        min_epoch_for_best if min_epoch_for_best is not None else cfg.MIN_EPOCH_FOR_BEST
    )
    log(f"[Start] {run_name} fold={fold} modalities={modalities} epochs={epochs}")
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_metrics = run_epoch(
            model, optimizer, train_loader, device, loss_cls_func, epoch
        )
        val_metrics = evaluate_split(
            model, val_loader, device, loss_cls_func, f"{run_name}-fold{fold}-val"
        )
        scheduler.step(val_metrics["overall"]["loss"])
        score = val_metrics["overall"][cfg.SELECTION_PRIMARY_METRIC]
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics["overall"],
                "seconds": time.time() - epoch_start,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        log(
            f"[{run_name}] fold={fold} epoch={epoch:03d} val_f1={score:.6f} val_auc={val_metrics['overall']['macro_auc_ovr']:.6f} lr={optimizer.param_groups[0]['lr']:.6g}"
        )
        if epoch >= min_best and np.isfinite(score) and (score > best_score):
            best_score = score
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for (k, v) in model.state_dict().items()
            }
            torch.save(best_state, out_dir / "best_model_state.pt")
        if best_epoch and epoch - best_epoch > cfg.EARLY_STOP_PATIENCE:
            break
    if best_state is None:
        raise RuntimeError(
            f"No valid best checkpoint was selected for {run_name} fold={fold}"
        )
    model.load_state_dict(best_state)
    val_final = evaluate_split(
        model, val_loader, device, loss_cls_func, f"{run_name}-fold{fold}-final-val"
    )
    save_history(history, out_dir)
    save_metrics("cv_val", val_final, out_dir)
    reloaded_model = model_factory().to(device)
    reloaded_state = torch.load(
        out_dir / "best_model_state.pt", map_location="cpu", weights_only=True
    )
    reloaded_model.load_state_dict(reloaded_state, strict=True)
    reloaded_val = evaluate_split(
        reloaded_model,
        val_loader,
        device,
        loss_cls_func,
        f"{run_name}-fold{fold}-checkpoint-reload-val",
    )
    saved_probs = np.asarray(val_final["probs"], dtype=np.float64)
    reloaded_probs = np.asarray(reloaded_val["probs"], dtype=np.float64)
    probability_max_abs = float(np.max(np.abs(saved_probs - reloaded_probs)))
    argmax_disagreements = int(
        np.count_nonzero(
            np.argmax(saved_probs, axis=1) != np.argmax(reloaded_probs, axis=1)
        )
    )
    macro_f1_abs_diff = abs(
        float(val_final["overall"]["macro_f1"])
        - float(reloaded_val["overall"]["macro_f1"])
    )
    checkpoint_qa = {
        "status": (
            "pass"
            if probability_max_abs <= 1e-05
            and argmax_disagreements == 0
            and (macro_f1_abs_diff <= 1e-12)
            else "fail"
        ),
        "run_name": run_name,
        "fold": fold,
        "split": "cv_val",
        "saved_macro_f1": float(val_final["overall"]["macro_f1"]),
        "reloaded_macro_f1": float(reloaded_val["overall"]["macro_f1"]),
        "macro_f1_abs_diff": macro_f1_abs_diff,
        "probability_max_abs": probability_max_abs,
        "argmax_disagreements": argmax_disagreements,
        "checkpoint_sha256": sha256(out_dir / "best_model_state.pt"),
    }
    (out_dir / "checkpoint_reload_qa.json").write_text(
        json.dumps(checkpoint_qa, indent=2), encoding="utf-8"
    )
    if checkpoint_qa["status"] != "pass":
        raise RuntimeError(f"Checkpoint reload QA failed: {checkpoint_qa}")
    del reloaded_model
    result = {
        "run_name": run_name,
        "fold": fold,
        "modalities": list(modalities),
        "max_epochs": epochs,
        "best_epoch": best_epoch,
        "lr": cfg.LR,
        "weight_decay": cfg.WEIGHT_DECAY,
        "batch_size": cfg.FIXED_BATCH_SIZE,
        "val_overall": val_final["overall"],
        "seed": cfg.SEED + fold,
        "external_test_evaluated": False,
        "checkpoint_reload_qa": checkpoint_qa,
        "out_dir": str(out_dir),
        "total_seconds": time.time() - start,
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    release_loader(train_loader)
    release_loader(val_loader)
    return result
