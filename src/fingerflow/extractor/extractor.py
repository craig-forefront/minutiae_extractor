import numpy as np
import pandas as pd

from .minutiae_net import MinutiaeNet
from .classify_net import ClassifyNet
from .core_net import CoreNet
from . import direction_sense, utils

# Smallest side, in px, of a print box that crop_to_print will crop to
MIN_CROP_SIZE = 128


def unclassified_minutiae_points(extracted_points):
    """The table that ClassifyNet returns, with NaN as the class of every minutia."""
    columns = ['x', 'y', 'angle', 'score', 'class']
    points = np.asarray(extracted_points, dtype=float)
    if points.size == 0:
        return pd.DataFrame([], columns=columns)

    table = pd.DataFrame(points[:, :4], columns=columns[:4])
    table['class'] = np.nan

    return table


class Extractor:
    """Finds the minutiae, and optionally their type and the core, of a fingerprint image.

    The networks are loaded once, when the object is made; call extract_minutiae for
    every image.

    Images have to be at about 500 ppi (pixels per inch). The minutiae networks work at
    that one scale and nothing here rescales the image, so resample images of any other
    resolution to 500 ppi before the call.

    Parameters:
        coarse_net_path: path of the CoarseNet weights (CoarseNet.h5). CoarseNet proposes
            the minutiae.
        fine_net_path: path of the FineNet weights (FineNet.h5). FineNet scores the image
            patch around every proposed minutia.
        classify_net_path: path of the ClassifyNet weights (ClassifyNet_6_classes.h5), or
            None. With None the minutiae are not classified: the 'class' column is NaN,
            and one network call per minutia is saved.
        core_net_path: path of the CoreNet weights (CoreNet.weights), or None. With None
            no core is detected and 'core' in the result is None.
            Neither of these two networks feeds back into the minutiae steps: position,
            angle, score and direction confidence are computed the same way with and
            without them. (Scores vary by less than 1e-6 from run to run in either case.)
        refine_direction: take the direction sense of the minutiae from the ridge pattern
            (module direction_sense) instead of the CoarseNet output, which is not
            stable. Default True. With False the result has no 'direction_confidence'
            column.
        image_dpi: scale of the direction step, and used by that step only. It does not
            rescale the image. Leave it at 500 for images at 500 ppi.
        crop_to_print: run the minutiae networks on the part of the image that holds the
            print only (utils.find_print_box) and report the coordinates in the full
            image. CoarseNet normalises its input with the mean and variance of the whole
            image, so a large empty margin changes what it sees of the print. If the box
            found is narrower or lower than MIN_CROP_SIZE (128 px), or no textured region
            is found, the whole image is used, without a message. Default False.
        reweight_score: give the FineNet score more weight. Every CoarseNet cell above
            candidate_threshold becomes a candidate, the score of a minutia is
            (1 - fine_net_weight) * CoarseNet + fine_net_weight * FineNet, and minutiae
            with a score of at least score_threshold are returned; an image can then come
            back with no minutiae. Without it (the default) the score is
            (4 * CoarseNet + FineNet) / 5 and the three numbers are not used.
        candidate_threshold, fine_net_weight, score_threshold: numbers from 0 to 1, see
            reweight_score.

    Raises ValueError for an image_dpi that is not a positive number and for one of the
    three numbers outside 0 to 1.
    """

    def __init__(self, coarse_net_path, fine_net_path, classify_net_path, core_net_path,
                 refine_direction=True, image_dpi=500, crop_to_print=False,
                 reweight_score=False, candidate_threshold=0.1, fine_net_weight=0.5,
                 score_threshold=0.55):
        if not isinstance(image_dpi, (int, float)) or not image_dpi > 0:
            raise ValueError(f'image_dpi has to be a positive number, got {image_dpi}')

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
        # None for a path switches that network off (see the class docstring)
        self.__classification_module = None
        if classify_net_path is not None:
            self.__classification_module = ClassifyNet(classify_net_path)
        self.__core_detection_module = None
        if core_net_path is not None:
            self.__core_detection_module = CoreNet(core_net_path)

    def extract_minutiae(self, image_data):
        """Extracts the minutiae, and the core if CoreNet was loaded, from one image.

        image_data: 8-bit (uint8) numpy array with dark ridges on a light background at
        about 500 ppi, for example the result of cv2.imread. Grayscale (2D), BGR and BGRA
        arrays are accepted; anything else raises ValueError.

        Returns a dictionary with two entries:
            'minutiae': pandas DataFrame with one row per minutia and the columns
                x, y: position in px in the image passed in (x is the column, y the row,
                    y pointing down), also with crop_to_print;
                angle: direction in radians, 0 to 2 pi; the direction vector is
                    (cos(angle), sin(angle)) in these axes;
                score: 0 to 1; its meaning depends on reweight_score (class docstring);
                class: 0 ending, 1 bifurcation, 2 fragment, 3 enclosure, 4 crossbar,
                    5 other; NaN if ClassifyNet was not loaded;
                direction_confidence: how clear the ridge pattern was that the direction
                    sense was taken from; absent if refine_direction is off.
            'core': pandas DataFrame with the columns x1, y1, x2, y2, score, w, h (one
                row per detected core box), or None if CoreNet was not loaded.
        """
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

        if self.__classification_module is not None:
            classified_points = self.__classification_module.classify_minutiae_points(
                preprocessed_image['original_image'],
                extracted_points)
        else:
            classified_points = unclassified_minutiae_points(extracted_points)

        if direction_confidence is not None:
            classified_points['direction_confidence'] = direction_confidence

        if x_offset or y_offset:
            classified_points['x'] = classified_points['x'] + x_offset
            classified_points['y'] = classified_points['y'] + y_offset

        core_coordinates = None
        if self.__core_detection_module is not None:
            core_coordinates = self.__core_detection_module.detect_fingerprint_core(image_data)

        return {'core': core_coordinates, 'minutiae': classified_points}
