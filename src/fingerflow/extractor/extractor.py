from .minutiae_net import MinutiaeNet
from .classify_net import ClassifyNet
from .core_net import CoreNet
from . import direction_sense, utils


class Extractor:
    def __init__(self, coarse_net_path, fine_net_path, classify_net_path, core_net_path,
                 refine_direction=True, image_dpi=500):
        # refine_direction: take the direction sense of the minutiae from the ridge pattern
        # instead of the CoarseNet output, which is not stable. image_dpi is used only by
        # this step.
        if not isinstance(image_dpi, (int, float)) or not image_dpi > 0:
            raise ValueError(f'image_dpi has to be a positive number, got {image_dpi}')

        self.__refine_direction = refine_direction
        self.__image_dpi = image_dpi
        self.__extraction_module = MinutiaeNet(coarse_net_path, fine_net_path)
        self.__classification_module = ClassifyNet(classify_net_path)
        self.__core_detection_module = CoreNet(core_net_path)

    def extract_minutiae(self, image_data):
        image_data = utils.normalize_image_data(image_data)

        preprocessed_image = utils.preprocess_image_data(image_data)

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

        core_coordinates = self.__core_detection_module.detect_fingerprint_core(image_data)

        return {'core': core_coordinates, 'minutiae': classified_points}
