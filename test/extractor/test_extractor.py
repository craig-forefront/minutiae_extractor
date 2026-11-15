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
        image = dislocation_image(self.direction, period=3.5)
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


def image_with_margin(direction, top=120, left=180):
    """White 480 x 480 px image with one 201 x 201 px print; its minutia is at the centre."""
    image = np.full((480, 480), 255, dtype=np.uint8)
    image[top:top + 201, left:left + 201] = dislocation_image(direction)

    return image, (left + 100, top + 100)


class ExtractorCropToPrintTest(unittest.TestCase):
    def setUp(self):
        self.direction = np.radians(40)
        self.image, self.minutia = image_with_margin(self.direction)
        self.box = extractor_module.utils.find_print_box(self.image)

    def run_extractor(self, **kwargs):
        """Returns the result and the images that the minutiae and the core step were given.

        The stub reports one minutia, at the position of the real one in the image it gets.
        """
        seen = {}

        def extract_points(image, original_image, **_kwargs):
            seen['image'], seen['original_image'] = image, original_image
            x_offset, y_offset = self.box[:2] if kwargs.get('crop_to_print') else (0, 0)
            return np.array([[self.minutia[0] - x_offset, self.minutia[1] - y_offset,
                              self.direction + np.pi, 0.9, 0.]])

        extractor = build_extractor(None, **kwargs)
        modules = {name.split('__')[-1]: value for name, value in vars(extractor).items()}
        modules['extraction_module'].extract_minutiae_points.side_effect = extract_points
        result = extractor.extract_minutiae(self.image)
        seen['core'] = modules['core_detection_module'].detect_fingerprint_core.call_args[0][0]

        return result['minutiae'], seen

    def test_whole_image_is_used_by_default(self):
        minutiae, seen = self.run_extractor()

        self.assertEqual(seen['original_image'].shape, (480, 480))
        self.assertEqual((minutiae['x'][0], minutiae['y'][0]), self.minutia)

    def test_networks_get_the_print_and_coordinates_are_of_the_whole_image(self):
        minutiae, seen = self.run_extractor(crop_to_print=True)
        x_start, y_start, x_end, y_end = self.box

        self.assertLess(x_end - x_start, 300)
        self.assertEqual(seen['original_image'].shape, (y_end - y_start, x_end - x_start))
        np.testing.assert_array_equal(
            seen['original_image'], self.image[y_start:y_end, x_start:x_end])
        self.assertEqual(seen['image'].shape[0] % 8, 0)
        self.assertEqual((minutiae['x'][0], minutiae['y'][0]), self.minutia)
        self.assertEqual(list(minutiae.columns), COLUMNS + ['direction_confidence'])
        self.assertLess(circular_distance(minutiae['angle'][0], self.direction), np.radians(10))
        # The core step still gets the whole image
        self.assertEqual(seen['core'].shape, (480, 480, 3))

    def test_no_minutiae(self):
        extractor = build_extractor(np.array([]), crop_to_print=True)
        minutiae = extractor.extract_minutiae(self.image)['minutiae']

        self.assertEqual(len(minutiae), 0)

    def test_image_without_print_is_not_cropped(self):
        self.image = np.full((480, 480), 255, dtype=np.uint8)
        self.box = (0, 0, 480, 480)
        _, seen = self.run_extractor(crop_to_print=True, refine_direction=False)

        self.assertEqual(seen['original_image'].shape, (480, 480))

    def test_small_print_box_is_not_cropped_to(self):
        self.image = np.full((480, 480), 255, dtype=np.uint8)
        self.image[200:260, 200:260] = dislocation_image(0.0)[:60, :60]
        self.box = (0, 0, 480, 480)
        self.minutia = (230, 230)
        minutiae, seen = self.run_extractor(crop_to_print=True, refine_direction=False)

        self.assertEqual(seen['original_image'].shape, (480, 480))
        self.assertEqual((minutiae['x'][0], minutiae['y'][0]), self.minutia)

