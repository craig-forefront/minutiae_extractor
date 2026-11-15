from .minutiae_net import MinutiaeNet
from .classify_net import ClassifyNet
from .core_net import CoreNet
from . import direction_sense, utils

# Smallest side, in px, of a print box that crop_to_print will crop to
MIN_CROP_SIZE = 128


class Extractor:
    def __init__(self, coarse_net_path, fine_net_path, classify_net_path, core_net_path,
                 refine_direction=True, image_dpi=500, crop_to_print=False,
                 reweight_score=False, candidate_threshold=0.1, fine_net_weight=0.5,
                 score_threshold=0.55):
        # refine_direction: take the direction sense of the minutiae from the ridge pattern
        # instead of the CoarseNet output, which is not stable. image_dpi is used only by
        # this step.
        if not isinstance(image_dpi, (int, float)) or not image_dpi > 0:
            raise ValueError(f'image_dpi has to be a positive number, got {image_dpi}')

        # crop_to_print: run the minutiae networks on the part of the image that holds the
        # print only (utils.find_print_box) and report the coordinates in the full image.
        # CoarseNet normalises its input with the mean and variance of the whole image, so
        # a large empty margin changes what it sees of the print.
        #
        # reweight_score: give the FineNet score more weight. Every CoarseNet cell above
        # candidate_threshold becomes a candidate, the score of a minutia is
        # (1 - fine_net_weight) * CoarseNet + fine_net_weight * FineNet, and minutiae with a
        # score of at least score_threshold are returned. Without it the score is
        # (4 * CoarseNet + FineNet) / 5 and the three numbers are not used.
        for name, value in (('candidate_threshold', candidate_threshold),
                            ('fine_net_weight', fine_net_weight),
                            ('score_threshold', score_threshold)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not 0 <= value <= 1:
                raise ValueError(f'{name} has to be a number from 0 to 1, got {value}')

        self.__crop_to_print = bool(crop_to_print)
        self.__score_reweighting = None
        if reweight_score:
            self.__score_reweighting = (candidate_threshold, fine_net_weight, score_threshold)

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

        if self.__score_reweighting is None:
            extracted_points = self.__extraction_module.extract_minutiae_points(
                preprocessed_image['image'],
                preprocessed_image['original_image'])
        else:
            extracted_points = self.__extraction_module.extract_minutiae_points(
                preprocessed_image['image'],
                preprocessed_image['original_image'],
                score_reweighting=self.__score_reweighting)

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
