import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import cv_config as cfg
from cv_data import read_manifest, ManifestDataset, BatchSpec
from evaluation import load_model, freeze_cnn
from representation_geometry import balanced_indices, embed
from workflow import selected


def specifications():
    chosen = selected()
    paths = {
        m: cfg.OUTPUT_ROOT / "cnn/single" / m / b / "fold_0"
        for (m, b) in chosen.items()
    }
    paths["Fusion"] = cfg.OUTPUT_ROOT / "cnn/fusion/BFT+BFR+DFR/fold_0"
    return paths


def single_gradcam_checkpoints():
    paths = {}
    for modality, backbone in selected().items():
        root = cfg.OUTPUT_ROOT / 'cnn/single' / modality / backbone
        candidates = []
        for fold in range(3):
            path = root / f'fold_{fold}'
            result = json.loads((path / 'experiment_result.json').read_text())
            candidates.append((-float(result['val_overall']['macro_f1']), fold, path))
        paths[modality] = min(candidates)[2]
    return paths


def single_gradcam(model, inputs, target):
    captured = {}

    def before_pool(_module, args):
        captured['activation'] = args[0]
        args[0].retain_grad()

    hook = model.branch.gap.register_forward_pre_hook(before_pool)
    model.zero_grad(set_to_none=True)
    try:
        logits = model(inputs)['cls_pred']
        logits[0, target].backward()
        activation = captured['activation']
        gradients = activation.grad
        if gradients is None or not torch.isfinite(gradients).all():
            raise RuntimeError('Invalid Grad-CAM gradients')
        raw = torch.relu((gradients.mean((2, 3), keepdim=True) * activation).sum(1))[0]
        raw = raw.detach().cpu().numpy()
        shifted = raw - raw.min()
        normalized = shifted / shifted.max() if shifted.max() > 0 else np.zeros_like(raw)
        return raw, normalized, logits.detach().softmax(1)[0].cpu().numpy()
    finally:
        hook.remove()


def main(task, extra):
    p = argparse.ArgumentParser()
    p.add_argument("--sample-id")
    args = p.parse_args(extra)
    freeze_cnn()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [r for r in read_manifest() if r["original_split"] == "test"]
    out = cfg.OUTPUT_ROOT / "interpretation"
    out.mkdir(parents=True, exist_ok=True)
    if task == "representations":
        summary = []
        for name, path in specifications().items():
            (model, mods) = load_model(path, device)
            loader = DataLoader(
                ManifestDataset(rows, BatchSpec(mods, False)),
                batch_size=16,
                shuffle=False,
                num_workers=1,
            )
            features = []
            with torch.inference_mode():
                for batch in loader:
                    output = model(batch["images"].to(device))
                    head = model.classifier if len(mods) == 1 else model.fusion_head
                    features.append(
                        head.net[:3](output["feature_fusion"]).cpu().numpy()
                    )
            values = np.concatenate(features)
            labels = np.asarray([int(r["label"]) for r in rows])
            ids = np.asarray([r["sample_id"] for r in rows])
            idx = balanced_indices(labels)
            (xy, quality) = embed(values[idx], labels[idx])
            np.savez_compressed(
                out / f"{name}_features.npz",
                features=values,
                label=labels,
                sample_id=ids,
            )
            pd.DataFrame(
                {
                    "sample_id": ids[idx],
                    "label": labels[idx],
                    "x": xy[:, 0],
                    "y": xy[:, 1],
                }
            ).to_csv(out / f"{name}_tsne.csv", index=False)
            summary.append({"representation": name, **quality})
        pd.DataFrame(summary).to_csv(out / "geometry.csv", index=False)
    else:
        chosen = [r for r in rows if r["sample_id"] == args.sample_id]
        if len(chosen) != 1:
            raise ValueError("Provide one test sample ID from the generated manifest")
        payload = {}
        for mod, path in single_gradcam_checkpoints().items():
            model, mods = load_model(path, device)
            if mods != (mod,):
                raise ValueError(f'Expected an independent {mod} model: {path}')
            sample = ManifestDataset(chosen, BatchSpec(mods, False))[0]
            inputs = sample['images'].unsqueeze(0).to(device)
            with torch.no_grad():
                baseline = model(inputs)['cls_pred'].softmax(1)[0].cpu().numpy()
            raw, normalized, probabilities = single_gradcam(model, inputs, int(chosen[0]['label']))
            np.testing.assert_allclose(probabilities, baseline, rtol=0, atol=1e-6)
            payload[mod] = normalized
            payload[f'{mod}_raw_positive_cam'] = raw
            payload[f'{mod}_probabilities'] = probabilities
            payload[f'{mod}_checkpoint'] = np.asarray(str(path))
            del model
        np.savez_compressed(out / f"{args.sample_id}_gradcam.npz", **payload)
