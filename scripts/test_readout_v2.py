"""Fast numerical checks for the offline readout benchmark (no connectome needed)."""
import unittest
import numpy as np
from scripts.compare_readout_v2 import features, fit, predict, metrics


class ReadoutTests(unittest.TestCase):
    def test_legacy_features_match_original_aggregation(self):
        raw = np.arange(10, dtype=float)[None, :]
        actual = features(raw, np.zeros_like(raw), 'legacy3')
        np.testing.assert_allclose(actual, [[-2, -16, -2]])

    def test_centered_neutral_is_zero(self):
        raw = np.ones((5,10))*3
        np.testing.assert_array_equal(features(raw, raw, 'centered10'), np.zeros_like(raw))

    def test_linear_map_recovery(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(100,10))
        weights = rng.normal(size=(10,3))
        y = x@weights + np.array([1.,2.,3.])
        model = fit(x, y, 0.)
        xt = rng.normal(size=(30,10))
        np.testing.assert_allclose(predict(model, xt), xt@weights+[1.,2.,3.], atol=1e-10)

    def test_constant_channels_remain_finite(self):
        model = fit(np.ones((20,10)), np.ones((20,3)), 1.)
        np.testing.assert_allclose(predict(model, np.ones((4,10))), np.ones((4,3)))

    def test_no_signal_reference_is_nonzero(self):
        y = np.eye(3)
        np.testing.assert_allclose(metrics(y,np.zeros_like(y))['rmse_rad_s'], np.sqrt(1/3))


if __name__ == '__main__':
    unittest.main()
