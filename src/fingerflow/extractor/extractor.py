from .minutiae_net import MinutiaeNet
from .classify_net import ClassifyNet
from .core_net import CoreNet
from . import direction_sense, utils

# Smallest side, in px, of a print box that crop_to_print will crop to
MIN_CROP_SIZE = 128


class Extractor:
    def __init__(self, coarse_net_path, fine_net_path, classify_net_path, core_net_path,
                 refine_direction=True, image_dpi=500, crop_to_print=False):
        # refine_direction: take the direction sense of the minutiae from the ridge pattern
        # instead of the CoarseNet output, which is not stable. image_dpi is used only by
        # this step.
        if not isinstance(image_dpi, (int, float)) or not image_dpi > 0:
            raise ValueError(f'image_dpi has to be a positive number, got {image_dpi}')

        # crop_to_print: run the minutiae networks on the part of the image that holds the
        # print only (utils.find_print_box) and report the coordinates in the full image.
        # CoarseNet normalises its input with the mean and variance of the whole image, so
        # a large empty margin changes what it sees of the print.
        self.__crop_to_print = bool(crop_to_print)

        self.__refine_direction = refine_direction
        self.__image_dpi = image_dpi
        self.__extraction_module = MinutiaeNet(coarse_net_path, fine_net_path)
        self.__classification_module = ClassifyNet(classify_net_path)
        self.__core_detection_module = CoreNet(core_net_path)

    def extract_minutiae(self, image_data):
        image_data = utils.normalize_image_data(image_data)

        # The minutiae steps work on minutiae_image_data, which is the whole image unless
        # crop_to_print is on; (x_offset, y_offset) is its top-left corner in the whole image
        minutiae_image_data = image_data
        x_offset, y_offset = 0, 0
        if self.__crop_to_print:
            x_offset, y_offset, x_end, y_end = utils.find_print_box(
                utils.preprocess_image_data(image_data)['original_image'])
            if min(x_end - x_offset, y_end - y_offset) < MIN_CROP_SIZE:
                # Too little to work on (the orientation step alone uses 64 px windows)
                x_offset, y_offset = 0, 0
            else:
                minutiae_image_data = image_data[y_offset:y_end, x_offset:x_end]

        preprocessed_image = utils.preprocess_image_data(minutiae_image_data)

        extracted_points = self.__extraction_module.extract_minutiae_points(
            preprocessed_image['image'],
            preprocessed_image['original_image'])

        direction_confidence = None
        if self.__refine_direction:
            extracted_points, direction_confidence = direction_sense.refine_minutiae_direction(
                preprocessed_image['original_image'], extracted_points, self.__image_dpi)

        classified_points = self.__classification_module.classify_minutiae_points(
            preprocessed_image['original_image'],
            extracted_points)

        if direction_confidence is not None:
            classified_points['direction_confidence'] = direction_confidence

        if x_offset or y_offset:
            classified_points['x'] = classified_points['x'] + x_offset
            classified_points['y'] = classified_points['y'] + y_offset

        core_coordinates = self.__core_detection_module.detect_fingerprint_core(image_data)

        return {'core': core_coordinates, 'minutiae': classified_points}
