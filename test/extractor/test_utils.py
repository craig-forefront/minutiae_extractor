# pylint: disable=no-self-use
import unittest
import numpy as np

from src.fingerflow.extractor import utils


class PreprocessImageDataTest(unittest.TestCase):
    def test_correct_image_data(self):
        mock_input_data = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_input_data_size = np.array(mock_input_data.shape, dtype=np.int32) // 8 * 8

        mock_output_data = {
            'original_image': mock_input_data[:, :, -1],
            'image_size': mock_input_data_size[:-1],
            'image': mock_input_data[:mock_input_data_size[0], :mock_input_data_size[0], -1]
        }

        result_output_data = utils.preprocess_image_data(mock_input_data)

        np.testing.assert_array_equal(
            result_output_data.get('original_image'),
            mock_output_data.get('original_image'))

        np.testing.assert_array_equal(
            result_output_data.get('image_size'),
            mock_output_data.get('image_size'))

        np.testing.assert_array_equal(
            result_output_data.get('image'),
            mock_output_data.get('image'))


class NormalizeImageDataTest(unittest.TestCase):
    def test_bgr_image_is_unchanged(self):
        mock_input_data = np.arange(4 * 5 * 3, dtype=np.uint8).reshape((4, 5, 3))

        np.testing.assert_array_equal(utils.normalize_image_data(mock_input_data), mock_input_data)

    def test_grayscale_image_is_expanded(self):
        mock_input_data = np.arange(20, dtype=np.uint8).reshape((4, 5))
        mock_output_data = np.stack([mock_input_data] * 3, axis=-1)

        np.testing.assert_array_equal(
            utils.normalize_image_data(mock_input_data), mock_output_data)
        np.testing.assert_array_equal(
            utils.normalize_image_data(mock_input_data[:, :, np.newaxis]), mock_output_data)

    def test_alpha_channel_is_dropped(self):
        mock_input_data = np.arange(4 * 5 * 4, dtype=np.uint8).reshape((4, 5, 4))

        np.testing.assert_array_equal(
            utils.normalize_image_data(mock_input_data), mock_input_data[:, :, :3])

    def test_non_uint8_image_is_rejected(self):
        with self.assertRaises(ValueError):
            utils.normalize_image_data(np.zeros((4, 5, 3), dtype=np.float32))


class GetMinutiaePatchCoordinateTest(unittest.TestCase):
    # patch should be aligned to the patch center
    def test_coordinates_without_shift(self):
        mock_patch_center = 50
        mock_image_size = 100

        mock_output_coordinates = (28, 72)

        result_output_coordinates = utils.get_minutiae_patch_coordinate(
            mock_patch_center, mock_image_size)

        self.assertTupleEqual(result_output_coordinates, mock_output_coordinates)

    # space between image start edge and patch center is smaller than PATCH_MINU_RATIO
    # patch should be aligned to the image start
    def test_coordinates_with_start_shift(self):
        mock_patch_center = 21
        mock_image_size = 100

        mock_output_coordinates = (0, 44)

        result_output_coordinates = utils.get_minutiae_patch_coordinate(
            mock_patch_center, mock_image_size)

        self.assertTupleEqual(result_output_coordinates, mock_output_coordinates)

    # space between image end edge and patch center is smaller than PATCH_MINU_RATIO
    # patch should be aligned to the end start
    def test_coordinates_with_end_shift(self):
        mock_patch_center = 79
        mock_image_size = 100

        mock_output_coordinates = (56, 100)

        result_output_coordinates = utils.get_minutiae_patch_coordinate(
            mock_patch_center, mock_image_size)

        self.assertTupleEqual(result_output_coordinates, mock_output_coordinates)


class GetMinutiaePatchTest(unittest.TestCase):
    def test_patch_without_shift(self):
        mock_input_image = np.array(
            np.meshgrid(np.arange(100),
                        np.arange(100))).reshape(
            (100, 100, 2))
        mock_patch_center = (50, 50)

        mock_output_patch = mock_input_image[28:72, 28:72]

        result_output_patch = utils.get_minutiae_patch(
            mock_patch_center[0], mock_patch_center[1], mock_input_image)

        np.testing.assert_array_equal(result_output_patch, mock_output_patch)

    def test_patch_with_top_shift(self):
        mock_input_image = np.array(
            np.meshgrid(np.arange(100),
                        np.arange(100))).reshape(
            (100, 100, 2))
        mock_patch_center = (50, 21)

        mock_output_patch = mock_input_image[0:44, 28:72]

        result_output_patch = utils.get_minutiae_patch(
            mock_patch_center[0], mock_patch_center[1], mock_input_image)

        np.testing.assert_array_equal(result_output_patch, mock_output_patch)

    def test_patch_with_bottom_shift(self):
        mock_input_image = np.array(
            np.meshgrid(np.arange(100),
                        np.arange(100))).reshape(
            (100, 100, 2))
        mock_patch_center = (50, 79)

        mock_output_patch = mock_input_image[56:100, 28:72]

        result_output_patch = utils.get_minutiae_patch(
            mock_patch_center[0], mock_patch_center[1], mock_input_image)

        np.testing.assert_array_equal(result_output_patch, mock_output_patch)

    def test_patch_with_left_shift(self):
        mock_input_image = np.array(
            np.meshgrid(np.arange(100),
                        np.arange(100))).reshape(
            (100, 100, 2))
        mock_patch_center = (21, 50)

        mock_output_patch = mock_input_image[28:72, 0:44]

        result_output_patch = utils.get_minutiae_patch(
            mock_patch_center[0], mock_patch_center[1], mock_input_image)

        np.testing.assert_array_equal(result_output_patch, mock_output_patch)

    def test_patch_with_right_shift(self):
        mock_input_image = np.array(
            np.meshgrid(np.arange(100),
                        np.arange(100))).reshape(
            (100, 100, 2))
        mock_patch_center = (79, 50)

        mock_output_patch = mock_input_image[28:72, 56:100]

        result_output_patch = utils.get_minutiae_patch(
            mock_patch_center[0], mock_patch_center[1], mock_input_image)

        np.testing.assert_array_equal(result_output_patch, mock_output_patch)

    def test_patch_with_top_left_shift(self):
        mock_input_image = np.array(
            np.meshgrid(np.arange(100),
                        np.arange(100))).reshape(
            (100, 100, 2))
        mock_patch_center = (21, 21)

        mock_output_patch = mock_input_image[0:44, 0:44]

        result_output_patch = utils.get_minutiae_patch(
            mock_patch_center[0], mock_patch_center[1], mock_input_image)

        np.testing.assert_array_equal(result_output_patch, mock_output_patch)

    def test_patch_with_top_right_shift(self):
        mock_input_image = np.array(
            np.meshgrid(np.arange(100),
                        np.arange(100))).reshape(
            (100, 100, 2))
        mock_patch_center = (79, 21)

        mock_output_patch = mock_input_image[0:44, 56:100]

        result_output_patch = utils.get_minutiae_patch(
            mock_patch_center[0], mock_patch_center[1], mock_input_image)

        np.testing.assert_array_equal(result_output_patch, mock_output_patch)

    def test_patch_with_bottom_left_shift(self):
        mock_input_image = np.array(
            np.meshgrid(np.arange(100),
                        np.arange(100))).reshape(
            (100, 100, 2))
        mock_patch_center = (21, 79)

        mock_output_patch = mock_input_image[56:100, 0:44]

        result_output_patch = utils.get_minutiae_patch(
            mock_patch_center[0], mock_patch_center[1], mock_input_image)

        np.testing.assert_array_equal(result_output_patch, mock_output_patch)

    def test_patch_with_bottom_right_shift(self):
        mock_input_image = np.array(
            np.meshgrid(np.arange(100),
                        np.arange(100))).reshape(
            (100, 100, 2))
        mock_patch_center = (79, 79)

        mock_output_patch = mock_input_image[56:100, 56:100]

        result_output_patch = utils.get_minutiae_patch(
            mock_patch_center[0], mock_patch_center[1], mock_input_image)

        np.testing.assert_array_equal(result_output_patch, mock_output_patch)

    def test_patch_on_non_square_image(self):
        mock_input_image = np.arange(200 * 100).reshape((200, 100))
        mock_patch_center = (30, 150)

        mock_output_patch = mock_input_image[128:172, 8:52]

        result_output_patch = utils.get_minutiae_patch(
            mock_patch_center[0], mock_patch_center[1], mock_input_image)

        np.testing.assert_array_equal(result_output_patch, mock_output_patch)


class ResizeMinutiaePatchtest(unittest.TestCase):
    def test_correct_patch_image_data(self):
        mock_input_data = np.zeros((44, 44))

        mock_output_patch = np.zeros((1, 224, 224, 3))

        result_output_patch = utils.resize_minutiae_patch(mock_input_data)

        np.testing.assert_array_equal(result_output_patch, mock_output_patch)


class FindPrintBoxTest(unittest.TestCase):
    @staticmethod
    def textured(height, width):
        """Stripes with a 10 px period, like ridges."""
        return np.tile((127 + 100 * np.sin(np.arange(width) * 2 * np.pi / 10)).astype(np.uint8),
                       (height, 1))

    def assert_box_close(self, box, expected, tolerance=8):
        for value, expected_value in zip(box, expected):
            self.assertLessEqual(abs(value - expected_value), tolerance)

    def test_print_on_white_background(self):
        image = np.full((600, 500), 255, dtype=np.uint8)
        image[200:400, 150:350] = self.textured(200, 200)

        self.assert_box_close(utils.find_print_box(image), (134, 184, 366, 416))
        self.assert_box_close(utils.find_print_box(image, margin=0), (150, 200, 350, 400))

    def test_box_contains_the_print_and_is_clipped_to_the_image(self):
        image = np.full((300, 300), 255, dtype=np.uint8)
        image[:150, 100:] = self.textured(150, 200)
        x_start, y_start, x_end, y_end = utils.find_print_box(image)

        self.assertEqual((y_start, x_end), (0, 300))
        self.assertLessEqual(x_start, 100)
        self.assertGreaterEqual(y_end, 150)

    def test_blank_image_gives_the_whole_image(self):
        self.assertEqual(utils.find_print_box(np.full((300, 200), 255, dtype=np.uint8)),
                         (0, 0, 200, 300))

    def test_print_filling_the_image_gives_the_whole_image(self):
        self.assertEqual(utils.find_print_box(self.textured(300, 200)), (0, 0, 200, 300))

    def test_several_prints_are_all_kept(self):
        image = np.full((400, 900), 255, dtype=np.uint8)
        image[100:300, 100:250] = self.textured(200, 150)
        image[120:280, 600:750] = self.textured(160, 150)

        self.assert_box_close(utils.find_print_box(image), (84, 84, 766, 316))

    def test_small_mark_far_from_the_print_is_ignored(self):
        image = np.full((400, 900), 255, dtype=np.uint8)
        image[100:300, 100:250] = self.textured(200, 150)
        image[150:180, 700:730] = self.textured(30, 30)

        self.assert_box_close(utils.find_print_box(image), (84, 84, 266, 316))

    def test_dark_background(self):
        image = np.zeros((600, 500), dtype=np.uint8)
        image[200:400, 150:350] = self.textured(200, 200)

        self.assert_box_close(utils.find_print_box(image), (134, 184, 366, 416))
