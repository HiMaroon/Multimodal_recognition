import ast
from copy import deepcopy
import inspect
import os
import json
from pathlib import Path
import random
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/celloptics"))
import cv_config as cfg
import cv_train_utils as trainer
import protocol_guard as guard
import proxy_classification as proxy
import proxies
import workflow
from reproducibility import configure
from statistics_core import paired_rows, weights


def cell(field, suffix="1"):
    split = {"A": "train", "B": "val", "C": "test"}[field["partition"]]
    sid = field["field_id"] + "_cell" + suffix
    return dict(sample_id=sid, parent_field_id=field["field_id"],
                original_split=split, dataset_role="external_test" if split == "test" else "main",
                cell_type=field["cell_type"], label=cfg.CLASS_TO_INDEX[field["cell_type"]],
                fold=field["cv_fold"], **{c: sid + "_" + c + ".png" for c in guard.MODALITY_COLUMNS.values()})


class ProtocolIntegrityTest(unittest.TestCase):
    def test_cli_relaunch_with_fixed_hash_seed(self):
        environment = dict(os.environ)
        environment.pop("PYTHONHASHSEED", None)
        result = subprocess.run(
            [sys.executable, str(ROOT / "run.py"), "--help"],
            env=environment, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("proxy-evaluate", result.stdout)

    @classmethod
    def setUpClass(cls):
        cls.fields = json.loads((ROOT / "configs/fields.json").read_text())
        cls.dev = next(r for r in cls.fields if r["partition"] == "A" and r["cv_fold"] is not None)
        cls.test = next(r for r in cls.fields if r["partition"] == "C")

    def test_fields_valid(self):
        guard.validate_fields(self.fields)

    def test_duplicate_field_rejected(self):
        with self.assertRaises(ValueError):
            guard.validate_fields(self.fields + [self.fields[0]])

    def test_modality_swap_rejected(self):
        rows = deepcopy(self.fields[:1])
        rows[0]["images"]["BFT"], rows[0]["images"]["BFR"] = rows[0]["images"]["BFR"], rows[0]["images"]["BFT"]
        with self.assertRaises(ValueError):
            guard.validate_fields(rows)

    def test_test_field_fold_rejected(self):
        row = deepcopy(self.test)
        row["cv_fold"] = 0
        with self.assertRaises(ValueError):
            guard.validate_fields([row])

    def test_test_cell_in_development_rejected(self):
        with self.assertRaises(ValueError):
            guard.validate_cells([cell(self.test)], development_only=True)

    def test_cell_relabelled_as_development_rejected(self):
        row = cell(self.test)
        row.update(original_split="train", dataset_role="main", fold=0)
        with self.assertRaises(ValueError):
            guard.validate_cells([row], development_only=True)

    def test_duplicate_cell_rejected(self):
        row = cell(self.dev)
        with self.assertRaises(ValueError):
            guard.validate_cells([row, row])

    def test_parent_field_fold_change_rejected(self):
        row = cell(self.dev)
        row["fold"] = (row["fold"] + 1) % 3
        with self.assertRaises(ValueError):
            guard.validate_cells([row])

    def test_parent_overlap_rejected(self):
        with self.assertRaises(ValueError):
            guard.validate_training_rows([cell(self.dev, "1")], [cell(self.dev, "2")])

    def test_training_has_no_test_evaluation_interface(self):
        self.assertNotIn("evaluate_external_test", inspect.signature(trainer.train_fold).parameters)
        source = inspect.getsource(trainer.train_fold)
        self.assertNotIn("external_test_rows", source)
        self.assertNotIn("test_loader", source)

    def test_backbone_selection_ignores_test_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            for modality in cfg.MODALITIES:
                for index, backbone in enumerate(cfg.BACKBONES):
                    for fold in range(3):
                        root = out / "cnn/single" / modality / backbone / f"fold_{fold}"
                        root.mkdir(parents=True)
                        (root / "best_model_state.pt").write_bytes(b"test fixture")
                        guard.save_json(root / "checkpoint_reload_qa.json", {"status": "pass", "checkpoint_sha256": guard.sha256(root / "best_model_state.pt")})
                        guard.save_json(root / "architecture.json", {"modalities": [modality], "backbones": {modality: backbone}, "modality_columns": guard.MODALITY_COLUMNS})
                        guard.save_json(root / "experiment_result.json", {"val_overall": {"macro_f1": .9-index*.01, "macro_auc_ovr": .99}, "test_overall": {"macro_f1": index / 5}})
            with patch.object(cfg, "OUTPUT_ROOT", out), patch.object(cfg, "SUMMARY_DIR", out / "summaries"), patch.object(cfg, "SELECTED_BACKBONES_PATH", out / "selected.json"):
                workflow.select([])
                result = workflow.selected()
                self.assertEqual(set(result.values()), {next(iter(cfg.BACKBONES))})

    def test_tiny_training_repeats_and_checkpoint_qa(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Linear(3, 3)

            def forward(self, inputs):
                return {"cls_pred": self.head(inputs.mean((2, 3)))}

        pairs = [[f for f in self.fields if f["partition"] == "A" and f["cv_fold"] is not None and f["cell_type"] == c][:2] for c in cfg.CLASS_ORDER]
        fields = [pair[i] for i in range(2) for pair in pairs]
        rows = [cell(f) for f in fields]
        for r in rows:
            r.update(image_name=r["sample_id"] + ".png", mask_path=r["sample_id"] + "_mask.png")
        def image(path):
            return np.full((16, 16) if "_mask" in str(path) else (16, 16, 3), 255 if "_mask" in str(path) else 100, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp, patch.object(trainer, "rows_for_fold", side_effect=lambda fold, train: rows[:3] if train else rows[3:]), patch("cv_data.read_image", side_effect=image), patch.object(cfg, "NUM_WORKERS", 0), patch.object(cfg, "FIXED_BATCH_SIZE", 3), patch.object(cfg, "PRINT_EVERY_N_BATCHES", 0), patch.object(cfg, "USE_AMP", False), patch("torch.cuda.is_available", return_value=False):
            states = []
            for run in ("first", "second"):
                root = Path(temp) / run
                result = trainer.train_fold("BFR", 0, ("BFR",), Tiny, root, max_epochs=1, min_epoch_for_best=1)
                self.assertFalse(result["external_test_evaluated"])
                self.assertEqual(result["checkpoint_reload_qa"]["status"], "pass")
                states.append(torch.load(root / "best_model_state.pt", weights_only=True))
            for key in states[0]:
                torch.testing.assert_close(states[0][key], states[1][key], atol=0, rtol=0)

    def test_proxy_selection_does_not_load_test_features(self):
        with patch.object(proxy, "load_features", side_effect=RuntimeError("stop after split check")) as load:
            with patch.object(proxy, "require_development"):
                with self.assertRaisesRegex(RuntimeError, "split check"):
                    proxy.classify()
            load.assert_called_once_with(("train", "val"))

    def test_locks_block_development_and_changed_artifacts(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(cfg, "OUTPUT_ROOT", Path(temp)):
            path = Path(temp) / "model.pt"
            path.write_bytes(b"original")
            guard.freeze("cnn", [path])
            guard.freeze("cnn", [path])
            with self.assertRaises(RuntimeError):
                guard.require_development("cnn")
            path.write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                guard.freeze("cnn", [path])

    def test_control_seeds_are_identity_based(self):
        seeds = [proxies.control_seed("p05", f"cell{i}", .5, r) for i in range(32) for r in range(8)]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertEqual(proxies.control_seed("p05", "cell2", .5, 3), seeds[2 * 8 + 3])

    def test_representative_controls_repeat_and_match_rms(self):
        yy, xx = np.indices((128, 128))
        mask = (xx-64)**2 + (yy-64)**2 < 42**2
        image = np.random.default_rng(5).uniform(.1, .9, (128, 128, 3)).astype(np.float32)
        for code in ("p05", "p06", "p07"):
            full, _ = proxies.neutralize(image, mask, code)
            seed = proxies.control_seed(code, "fixture", 1., 0)
            a, qc = proxies.control(image, full, mask, 1., seed, code)
            b, _ = proxies.control(image, full, mask, 1., seed, code)
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(a[~mask], image[~mask])
            self.assertEqual(qc["status"], "pass")
            self.assertLessEqual(qc["rms_relative_error"], .02)

    def test_seeded_streams_repeat(self):
        configure(123)
        first = (random.random(), np.random.random(), torch.rand(5))
        configure(123)
        second = (random.random(), np.random.random(), torch.rand(5))
        self.assertEqual(first[:2], second[:2])
        torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())

    def test_cluster_bootstrap_repeats_and_preserves_strata_counts(self):
        labels = np.array([0, 0, 1, 2, 2, 2])
        a = weights(labels, 100, 7)
        np.testing.assert_array_equal(a, weights(labels, 100, 7))
        for label in range(3):
            np.testing.assert_array_equal(a[:, labels == label].sum(1), np.full(100, (labels == label).sum()))

    def test_paired_prediction_mismatch_rejected(self):
        frame = pd.DataFrame(dict(sample_id=["a", "b"], parent_field_id=["f1", "f2"], label=[0, 1]))
        bad = frame.copy()
        bad.loc[1, "label"] = 2
        with self.assertRaises(ValueError):
            paired_rows(frame, bad)

    def test_all_python_files_parse(self):
        for path in ROOT.rglob("*.py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


if __name__ == "__main__":
    unittest.main()
