import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/celloptics'))
from interpretation import single_gradcam, single_gradcam_checkpoints


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.branch = torch.nn.Module()
        self.branch.conv = torch.nn.Conv2d(3, 2, 1, bias=False)
        self.branch.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.head = torch.nn.Linear(2, 3, bias=False)

    def forward(self, x):
        features = self.branch.conv(x)
        return {'cls_pred': self.head(self.branch.gap(features).flatten(1))}


class SingleGradCAMTest(unittest.TestCase):
    def test_map_matches_linear_reference_and_removes_hook(self):
        torch.manual_seed(5)
        model = TinyClassifier().eval()
        x = torch.rand(1, 3, 7, 7)
        raw, norm, probs = single_gradcam(model, x, 1)
        with torch.no_grad():
            activation = model.branch.conv(x)
            reference = torch.relu((activation * model.head.weight[1].view(1,2,1,1) / 49).sum(1))[0].numpy()
            probability = model(x)['cls_pred'].softmax(1)[0].numpy()
        np.testing.assert_allclose(raw, reference, atol=1e-8)
        np.testing.assert_allclose(probs, probability, atol=1e-8)
        self.assertTrue(np.isfinite(norm).all())
        self.assertEqual(len(model.branch.gap._forward_pre_hooks), 0)

    def test_checkpoint_selection_uses_validation_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for fold, score in enumerate([.90, .95, .95]):
                path = root / 'cnn/single/BFR/mobilenet_v2' / f'fold_{fold}'
                path.mkdir(parents=True)
                (path/'experiment_result.json').write_text(json.dumps({
                    'val_overall':{'macro_f1':score},
                    'test_overall':{'macro_f1':1.0-fold*.1}}))
            with patch('interpretation.cfg.OUTPUT_ROOT', root), patch(
                    'interpretation.selected', return_value={'BFR':'mobilenet_v2'}):
                self.assertEqual(single_gradcam_checkpoints()['BFR'].name, 'fold_1')


if __name__ == '__main__':
    unittest.main()
