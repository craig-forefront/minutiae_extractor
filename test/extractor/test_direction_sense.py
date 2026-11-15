import unittest
import numpy as np

from src.fingerflow.extractor import direction_sense


def circular_distance(first, second):
    return abs(np.angle(np.exp(1j * (first - second))))


def dislocation_image(direction, period=10.0, polarity=1, size=201, center=None):
    """Ridge pattern with a single minutia whose conventional direction is `direction`.

    The extra ridge lies on the side the direction points to.
    """
    center_x, center_y = center if center is not None else (size // 2, size // 2)
    rows, cols = np.mgrid[0:size, 0:size].astype(float)
    cols -= center_x
    rows -= center_y
    u = cols * np.cos(direction) + rows * np.sin(direction)
    v = -cols * np.sin(direction) + rows * np.cos(direction)
    pattern = np.cos(2 * np.pi * v / period + np.arctan2(v, u))

    return (127 + 100 * polarity * pattern).astype(np.uint8)


class RefineMinutiaeDirectionTest(unittest.TestCase):
    def test_sense_is_set_for_either_input_sense(self):
        for direction in np.radians([0, 40, 115, 200, 290]):
            image = dislocation_image(direction)

            for given in (direction, direction + np.pi):
                minutiae = np.array([[100., 100., given % (2 * np.pi), 0.9, 0.]])
                refined, confidence = direction_sense.refine_minutiae_direction(image, minutiae)

                self.assertLess(circular_distance(refined[0, 2], direction), np.radians(10))
                self.assertGreater(confidence[0], 0.5)

    def test_polarity_does_not_matter(self):
        direction = np.radians(70)

        for polarity in (1, -1):
            image = dislocation_image(direction, polarity=polarity)
            minutiae = np.array([[100., 100., direction + np.pi, 0.9, 0.]])
            refined, _ = direction_sense.refine_minutiae_direction(image, minutiae)

            self.assertLess(circular_distance(refined[0, 2], direction), np.radians(10))

    def test_position_error_is_tolerated(self):
        direction = np.radians(160)
        image = dislocation_image(direction)
        minutiae = np.array([[103., 98., direction + np.pi, 0.9, 0.]])
        refined, _ = direction_sense.refine_minutiae_direction(image, minutiae)

        self.assertLess(circular_distance(refined[0, 2], direction), np.radians(10))

    def test_wrong_orientation_is_replaced_by_measured_one(self):
        direction = np.radians(30)
        image = dislocation_image(direction)
        minutiae = np.array([[100., 100., direction + np.radians(60), 0.9, 0.]])
        refined, _ = direction_sense.refine_minutiae_direction(image, minutiae)

        self.assertLess(circular_distance(refined[0, 2], direction), np.radians(10))

    def test_close_orientation_is_kept(self):
        direction = np.radians(30)
        image = dislocation_image(direction)
        given = direction + np.radians(12)
        minutiae = np.array([[100., 100., given + np.pi, 0.9, 0.]])
        refined, _ = direction_sense.refine_minutiae_direction(image, minutiae)

        self.assertLess(circular_distance(refined[0, 2], given), 1e-9)

    def test_image_dpi_scales_the_filters(self):
        direction = np.radians(250)

        for period, image_dpi in ((3.5, 250), (40.0, 1000)):
            image = dislocation_image(direction, period=period, size=401)
            minutiae = np.array([[200., 200., direction + np.pi, 0.9, 0.]])
            refined, confidence = direction_sense.refine_minutiae_direction(
                image, minutiae, image_dpi=image_dpi)
            _, default_confidence = direction_sense.refine_minutiae_direction(image, minutiae)

            self.assertLess(circular_distance(refined[0, 2], direction), np.radians(10))
            self.assertGreater(confidence[0], 0.5)
            self.assertLess(default_confidence[0], confidence[0] / 2)

    def test_invalid_image_dpi(self):
        image = dislocation_image(0.0)
        minutiae = np.array([[100., 100., 0., 0.9, 0.]])

        for image_dpi in (0, -500, float('nan')):
            with self.assertRaises(ValueError):
                direction_sense.refine_minutiae_direction(image, minutiae, image_dpi=image_dpi)

    def test_off_centre_minutia_on_non_square_image(self):
        direction = np.radians(20)
        rows, cols = np.mgrid[0:160, 0:300].astype(float)
        cols -= 215
        rows -= 60
        u = cols * np.cos(direction) + rows * np.sin(direction)
        v = -cols * np.sin(direction) + rows * np.cos(direction)
        image = (127 + 100 * np.cos(2 * np.pi * v / 10 + np.arctan2(v, u))).astype(np.uint8)

        # 70 degrees off, so the orientation has to be looked up at the right pixel as well
        minutiae = np.array([[215., 60., direction + np.radians(70), 0.9, 0.]])
        refined, confidence = direction_sense.refine_minutiae_direction(image, minutiae)

        self.assertLess(circular_distance(refined[0, 2], direction), np.radians(10))
        self.assertGreater(confidence[0], 0.5)

    def test_guard_threshold(self):
        direction = np.radians(30)
        image = dislocation_image(direction)

        kept = direction + np.radians(11)
        refined, _ = direction_sense.refine_minutiae_direction(
            image, np.array([[100., 100., kept, 0.9, 0.]]))
        self.assertLess(circular_distance(refined[0, 2], kept), 1e-9)

        refined, _ = direction_sense.refine_minutiae_direction(
            image, np.array([[100., 100., direction + np.radians(20), 0.9, 0.]]))
        self.assertLess(circular_distance(refined[0, 2], direction), np.radians(10))

    def test_guard_across_the_wrap(self):
        # orientations 175 and 5 degrees are 10 degrees apart
        direction = np.radians(175)
        image = dislocation_image(direction)
        given = np.radians(5)
        refined, _ = direction_sense.refine_minutiae_direction(
            image, np.array([[100., 100., given, 0.9, 0.]]))

        self.assertLess(circular_distance(refined[0, 2], given + np.pi), 1e-9)

    def test_blank_image_has_no_confidence(self):
        image = np.full((120, 120), 200, dtype=np.uint8)
        refined, confidence = direction_sense.refine_minutiae_direction(
            image, np.array([[60., 60., 1.0, 0.9, 0.]]))

        self.assertTrue(np.isfinite(refined).all())
        self.assertLess(confidence[0], 0.05)

    def test_minutia_near_border(self):
        direction = np.radians(90)
        image = dislocation_image(direction, center=(6, 195))
        minutiae = np.array([[6., 195., direction, 0.9, 0.]])
        refined, confidence = direction_sense.refine_minutiae_direction(image, minutiae)

        self.assertLess(circular_distance(refined[0, 2], direction), np.radians(15))
        self.assertTrue(np.isfinite(confidence).all())

    def test_other_columns_and_input_are_untouched(self):
        image = dislocation_image(0.0)
        minutiae = np.array([[100., 100., np.pi, 0.9, 3.]])
        original = minutiae.copy()
        refined, _ = direction_sense.refine_minutiae_direction(image, minutiae)

        np.testing.assert_array_equal(minutiae, original)
        np.testing.assert_array_equal(refined[:, [0, 1, 3, 4]], original[:, [0, 1, 3, 4]])

    def test_no_minutiae(self):
        image = dislocation_image(0.0)
        refined, confidence = direction_sense.refine_minutiae_direction(image, np.array([]))

        self.assertEqual(refined.size, 0)
        self.assertEqual(confidence.size, 0)
