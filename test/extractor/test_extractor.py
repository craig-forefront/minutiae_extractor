import unittest
from unittest import mock
import numpy as np
import pandas as pd

from src.fingerflow.extractor import extractor as extractor_module
from .test_direction_sense import dislocation_image, circular_distance

COLUMNS = ['x', 'y', 'angle', 'score', 'class']


def build_extractor(points, **kwargs):
    """Extractor with the three networks replaced by stubs returning `points`."""
    def classify(_image, extracted):
        if extracted.size == 0:
            return pd.DataFrame([], columns=COLUMNS)
        return pd.DataFrame(np.array(extracted), columns=COLUMNS)

    with mock.patch.object(extractor_module, 'MinutiaeNet') as minutiae_net, \
            mock.patch.object(extractor_module, 'ClassifyNet') as classify_net, \
            mock.patch.object(extractor_module, 'CoreNet') as core_net:
        minutiae_net.return_value.extract_minutiae_points.return_value = points
        classify_net.return_value.classify_minutiae_points.side_effect = classify
        core_net.return_value.detect_fingerprint_core.return_value = pd.DataFrame()

        return extractor_module.Extractor('a', 'b', 'c', 'd', **kwargs)


class ExtractorDirectionTest(unittest.TestCase):
    def setUp(self):
        self.direction = np.radians(40)
        self.image = dislocation_image(self.direction)
        self.points = np.array([[100., 100., self.direction + np.pi, 0.9, 0.]])

    def test_direction_is_refined_by_default(self):
        minutiae = build_extractor(self.points).extract_minutiae(self.image)['minutiae']

        self.assertEqual(list(minutiae.columns), COLUMNS + ['direction_confidence'])
        self.assertLess(circular_distance(minutiae['angle'][0], self.direction), np.radians(10))
        self.assertGreater(minutiae['direction_confidence'][0], 0.5)

    def test_refinement_can_be_turned_off(self):
        minutiae = build_extractor(
            self.points, refine_direction=False).extract_minutiae(self.image)['minutiae']

        self.assertEqual(list(minutiae.columns), COLUMNS)
        self.assertAlmostEqual(minutiae['angle'][0], self.direction + np.pi)

    def test_image_dpi_is_passed_on(self):
        image = dislocation_image(self.direction, period=4.0)
        points = self.points
        minutiae = build_extractor(points, image_dpi=250).extract_minutiae(image)['minutiae']
        default = build_extractor(points).extract_minutiae(image)['minutiae']

        self.assertGreater(minutiae['direction_confidence'][0], 0.5)
        self.assertLess(default['direction_confidence'][0], 0.25)

    def test_no_minutiae(self):
        minutiae = build_extractor(np.array([])).extract_minutiae(self.image)['minutiae']

        self.assertEqual(len(minutiae), 0)
        self.assertEqual(list(minutiae.columns), COLUMNS + ['direction_confidence'])

    def test_invalid_image_dpi(self):
        for image_dpi in (0, -1, None, 'high'):
            with self.assertRaises(ValueError):
                build_extractor(self.points, image_dpi=image_dpi)
