import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/celloptics'))
sys.dont_write_bytecode = True
from cv_data import BatchSpec, ManifestDataset


class ModalityIdentityTest(unittest.TestCase):
    def test_backbone_identity(self):
        protocol = json.loads((ROOT / 'configs/protocol.json').read_text())
        self.assertEqual(protocol['selected_backbones'], {
            'BFR': 'mobilenet_v2', 'BFT': 'densenet121', 'DFR': 'densenet121'})

    def test_raw_slots_and_tensor_order(self):
        row = dict(t_path='T.png', rb_path='RB.png', rd_path='RD.png',
                   mask_path='mask.png', label='0', image_name='cell.png',
                   sample_id='cell', cell_type='MCF7')
        values = {'T.png': 51, 'RB.png': 102, 'RD.png': 153, 'mask.png': 255}
        def read(path):
            shape = (128,128) if path.name == 'mask.png' else (128,128,3)
            return np.full(shape, values[path.name], dtype=np.uint8)
        with patch('cv_data.read_image', side_effect=read):
            for modalities in [('BFR',), ('BFT',), ('DFR',), ('BFR','BFT','DFR')]:
                result = ManifestDataset([row], BatchSpec(modalities))[0]
                for index, modality in enumerate(modalities):
                    expected = {'BFR':.2, 'BFT':.4, 'DFR':.6}[modality]
                    np.testing.assert_allclose(result['images'][3*index:3*index+3], expected)

    def test_field_names_match_keys(self):
        fields = json.loads((ROOT / 'configs/fields.json').read_text())
        for field in fields:
            for modality, name in field['images'].items():
                self.assertIn('-' + modality + '-', name)


if __name__ == '__main__':
    unittest.main()
