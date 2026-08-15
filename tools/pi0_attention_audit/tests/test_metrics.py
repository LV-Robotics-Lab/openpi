import unittest

import numpy as np

from pi0_attention_audit.metrics import normalize_heatmap, region_fraction, region_mean


class MetricsTest(unittest.TestCase):
    def test_normalize_constant_map_is_zero(self) -> None:
        result = normalize_heatmap(np.ones((2, 2), dtype=np.float32))
        np.testing.assert_array_equal(result, np.zeros((2, 2), dtype=np.float32))

    def test_region_fraction_uses_raw_mass(self) -> None:
        values = np.asarray([[1.0, 1.0], [2.0, 6.0]])
        self.assertAlmostEqual(region_fraction(values, (0.5, 0.5, 1.0, 1.0)), 0.6)

    def test_region_mean_uses_selected_cells(self) -> None:
        values = np.asarray([[0.0, 0.25], [0.5, 1.0]])
        self.assertAlmostEqual(region_mean(values, (0.0, 0.5, 1.0, 1.0)), 0.75)


if __name__ == "__main__":
    unittest.main()
