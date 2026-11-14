import unittest
import numpy as np

from src.fingerflow.extractor.MinutiaeNet.CoarseNet.coarse_net_model import fuse_minu_orientation


def circular_distance(first, second):
    return abs(np.angle(np.exp(1j * (first - second))))


def fuse(angle, orientation, mode):
    dir_map = np.full((4, 4), orientation, dtype=float)
    mnt = np.array([[20., 20., angle, 1., 0.]])
    fuse_minu_orientation(dir_map, mnt, mode=mode)

    return mnt[0, 2]


class FuseMinuOrientationTest(unittest.TestCase):
    def test_average_across_wrap(self):
        # 1 deg and 354 deg are 7 deg apart, the result has to sit between them
        result = fuse(np.radians(1), np.radians(-6), mode=3)

        self.assertLess(circular_distance(result, np.radians(357.5)), 1e-6)

    def test_average_uses_opposite_direction_when_closer(self):
        # orientation 86 deg is closer to 275 deg as 266 deg
        result = fuse(np.radians(275), np.radians(86), mode=3)

        self.assertLess(circular_distance(result, np.radians(270.5)), 1e-6)

    def test_average_stays_close_to_both_inputs(self):
        for angle in np.radians(np.arange(0, 360, 7)):
            for orientation in np.radians(np.arange(-90, 91, 9)):
                result = fuse(angle, orientation, mode=3)

                self.assertTrue(0 <= result < 2 * np.pi)
                self.assertLessEqual(circular_distance(result, angle), np.pi / 4 + 1e-6)

    def test_mode_1_takes_nearest_direction(self):
        result = fuse(np.radians(200), np.radians(10), mode=1)

        self.assertLess(circular_distance(result, np.radians(190)), 1e-6)

    def test_mode_2_keeps_angle(self):
        self.assertEqual(fuse(1.234, 0.5, mode=2), 1.234)

    def test_lookup_is_clamped_to_map(self):
        dir_map = np.zeros((2, 2))
        mnt = np.array([[100., 100., 0.1, 1., 0.]])

        fuse_minu_orientation(dir_map, mnt, mode=3)

        self.assertLess(circular_distance(mnt[0, 2], 0.05), 1e-6)
