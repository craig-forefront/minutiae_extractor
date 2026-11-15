import unittest
from unittest import mock
import numpy as np

from src.fingerflow.extractor import minutiae_net as minutiae_net_module
from .test_direction_sense import dislocation_image

# CoarseNet cell (row, column) -> CoarseNet score. A cell is 8 x 8 px. The cells are far
# enough apart that none suppresses another.
STRONG_CELLS = [(3, 3), (3, 10), (3, 17), (10, 3), (10, 10), (10, 17)]
CELL_SCORES = {cell: 0.9 for cell in STRONG_CELLS}
CELL_SCORES[(17, 3)] = 0.3
CELL_SCORES[(17, 10)] = 0.15


class FakeCoarseNet:
    """Returns the scores of CELL_SCORES for a 200 x 200 px image and an all-print mask."""

    def predict(self, image):
        rows, cols = image.shape[1] // 8, image.shape[2] // 8
        score = np.zeros((1, rows, cols, 1))
        for (row, col), value in CELL_SCORES.items():
            score[0, row, col, 0] = value

        segmentation = np.zeros((1, rows, cols, 1))
        orientation = np.zeros((1, rows, cols, 90))
        offset = np.zeros((1, rows, cols, 8))

        return [None] * 5 + [segmentation, orientation, offset, offset, score]


class FakeFineNet:
    def __init__(self, minutia_probability):
        self.minutia_probability = minutia_probability
        self.calls = 0

    def compile(self, **_kwargs):
        pass

    def predict(self, _patch):
        self.calls += 1
        return np.array([[self.minutia_probability, 1 - self.minutia_probability]])


def extract(fine_net_probability, **kwargs):
    fine_net = FakeFineNet(fine_net_probability)
    with mock.patch.object(minutiae_net_module.coarse_net_model, 'get_coarse_net_model',
                           return_value=FakeCoarseNet()), \
            mock.patch.object(minutiae_net_module.fine_net_model, 'get_fine_net_model',
                              return_value=fine_net):
        net = minutiae_net_module.MinutiaeNet('a', 'b')

    image = dislocation_image(0.0, size=200)
    points = net.extract_minutiae_points(image, image, **kwargs)

    return {(int(y) // 8, int(x) // 8): score for x, y, _, score, _ in points}, fine_net.calls


class ScoreReweightingTest(unittest.TestCase):
    def test_default_score_is_four_parts_coarse_net(self):
        scores, calls = extract(0.9)

        self.assertEqual(sorted(scores), sorted(STRONG_CELLS))
        self.assertEqual(calls, 6)
        for score in scores.values():
            self.assertAlmostEqual(score, (4 * 0.9 + 0.9) / 5)

    def test_default_ignores_a_low_fine_net_score(self):
        scores, _ = extract(0.1)

        self.assertEqual(sorted(scores), sorted(STRONG_CELLS))
        for score in scores.values():
            self.assertAlmostEqual(score, (4 * 0.9 + 0.1) / 5)

    def test_reweighting_adds_weak_cells_that_fine_net_confirms(self):
        scores, calls = extract(0.9, score_reweighting=(0.1, 0.5, 0.55))

        # 0.3 -> 0.5 * 0.3 + 0.5 * 0.9 = 0.6 is kept; 0.15 -> 0.525 is not
        self.assertEqual(sorted(scores), sorted(STRONG_CELLS + [(17, 3)]))
        self.assertEqual(calls, 8)
        self.assertAlmostEqual(scores[(17, 3)], 0.6)
        self.assertAlmostEqual(scores[(3, 3)], 0.9)

    def test_reweighting_drops_strong_cells_that_fine_net_rejects(self):
        scores, _ = extract(0.1, score_reweighting=(0.1, 0.5, 0.55))

        self.assertEqual(scores, {})

    def test_reweighting_numbers_are_used(self):
        scores, calls = extract(0.9, score_reweighting=(0.2, 0.25, 0.4))

        # The 0.15 cell is below the candidate threshold; 0.3 -> 0.75 * 0.3 + 0.25 * 0.9
        self.assertEqual(sorted(scores), sorted(STRONG_CELLS + [(17, 3)]))
        self.assertEqual(calls, 7)
        self.assertAlmostEqual(scores[(17, 3)], 0.45)

    def test_reweighting_does_not_lower_the_candidate_threshold(self):
        # The original behaviour lowers its threshold until five minutiae are found
        with mock.patch.dict(CELL_SCORES, {cell: 0.0 for cell in STRONG_CELLS[1:]}):
            default_scores, _ = extract(0.1)
            scores, _ = extract(0.1, score_reweighting=(0.1, 0.5, 0.55))

        self.assertIn((17, 10), default_scores)
        self.assertEqual(scores, {})
