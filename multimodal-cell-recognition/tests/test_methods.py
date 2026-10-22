import json
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/celloptics"))
sys.dont_write_bytecode = True
import proxies
from inference import adjust


class MethodsTest(unittest.TestCase):
    def test_registry_and_descriptors(self):
        self.assertEqual(list(proxies.BY_ID), [f"p{i:02d}" for i in range(1, 16)])
        self.assertEqual(sum(r["dimension"] for r in proxies.REGISTRY), 180)
        yy, xx = np.indices((128, 128))
        mask = (xx - 64) ** 2 + (yy - 64) ** 2 < 42**2
        image = (
            np.random.default_rng(5).uniform(0.1, 0.9, (128, 128, 3)).astype(np.float32)
        )
        for code, item in proxies.BY_ID.items():
            feature = proxies.descriptor(image, mask, code)
            self.assertEqual(len(feature), item["dimension"])
            self.assertTrue(np.isfinite(feature).all())
            result, qc = proxies.neutralize(image, mask, code)
            self.assertEqual(qc["status"], "pass")
            np.testing.assert_array_equal(result[~mask], image[~mask])

    def test_integral(self):
        doses = np.array([0, 0.25, 0.5, 0.75, 1.0])
        curve = np.array([0, 2, 3, 4, 5.0])
        self.assertAlmostEqual(
            float(np.trapz(curve, doses)),
            float(curve[1:] @ np.array([0.25, 0.25, 0.25, 0.125])),
        )

    def test_p01_uses_only_mask_interior(self):
        yy, xx = np.indices((96, 96))
        mask = (xx - 48) ** 2 + (yy - 48) ** 2 < 30**2
        image = np.full((96, 96, 3), 0.65, dtype=np.float32)
        expected = np.median(image[mask] @ np.array([0.299, 0.587, 0.114]))
        first = proxies.descriptor(image, mask, "p01")
        image[~mask] = 0.05
        np.testing.assert_array_equal(first, proxies.descriptor(image, mask, "p01"))
        self.assertAlmostEqual(float(first[0]), expected, places=6)

    def test_p01_intervention_ignores_exterior_reference(self):
        yy, xx = np.indices((96, 96))
        mask = (xx - 48) ** 2 + (yy - 48) ** 2 < 30**2
        first = np.full((96, 96, 3), 0.65, dtype=np.float32)
        second = first.copy()
        second[~mask] = 0.05
        a, qa = proxies.neutralize(first, mask, "p01")
        b, qb = proxies.neutralize(second, mask, "p01")
        self.assertEqual(qa["status"], "pass")
        self.assertEqual(qb["status"], "pass")
        np.testing.assert_array_equal(a[mask], b[mask])
        np.testing.assert_allclose(a[mask], 0, atol=1e-6)
        np.testing.assert_array_equal(b[~mask], second[~mask])

    def test_bh(self):
        np.testing.assert_allclose(adjust([0.01, 0.04, 0.03]), [0.03, 0.04, 0.04])
        self.assertEqual(len(adjust(np.linspace(0.001, 0.9, 45))), 45)

    def test_partitions(self):
        fields = json.loads((ROOT / "configs/fields.json").read_text())
        self.assertEqual(len(fields), 396)
        self.assertEqual(len({r["field_id"] for r in fields}), 396)
        for r in fields:
            self.assertEqual(set(r["images"]), {"BFR", "BFT", "DFR"})
            self.assertIn(r["partition"], ["A", "B", "C"])
            if r["partition"] == "C":
                self.assertIsNone(r["cv_fold"])
            else:
                self.assertIn(r["cv_fold"], [None, 0, 1, 2])


if __name__ == "__main__":
    unittest.main()
