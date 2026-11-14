"""Direction sense of minutiae measured from the image.

CoarseNet reports the ridge orientation of a minutia well, but which of the two opposite
directions it reports is not stable: the released weights flip it for a large share of
minutiae when the image is shifted by a few pixels. The functions here decide the sense
from the ridge pattern instead.

A minutia (ending or bifurcation, dark or bright ridges) is a dislocation of the ridge
pattern: along the ridge tangent there is exactly one more ridge on one side of it than
on the other. The conventional direction (ISO 19794-2, MinutiaeNet ground truth) points to
the side with more ridges: back along the ridge for an ending, into the valley between the
two branches for a bifurcation. That side is read as the sign of the phase vortex of a
complex Gabor response tuned to the local ridge orientation.

Angles use the convention of the extractor output: the direction vector is
(cos(angle), sin(angle)) in image axes, y pointing down.
"""
import cv2
import numpy as np

REFERENCE_DPI = 500.0
RIDGE_PERIODS = (7.0, 9.0, 11.0, 13.0)  # px at 500 dpi
WINDOW_SIGMA = 10.0  # px at 500 dpi
ORIENTATION_SIGMA = 6.0  # px at 500 dpi
CONTRAST_SIGMA = 8.0  # px at 500 dpi
ORIENTATION_GUARD = np.radians(25)


def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _to_gray(image):
    gray = np.asarray(image, dtype=np.float32)

    if gray.ndim == 3:
        gray = gray[..., :3].mean(axis=-1)

    return gray


def normalize_contrast(image, scale=1.0):
    """Local mean and variance normalisation, which makes the result polarity and gain free."""
    smooth = cv2.GaussianBlur(_to_gray(image), (0, 0), 1.0)
    mean = cv2.GaussianBlur(smooth, (0, 0), CONTRAST_SIGMA * scale)
    variance = cv2.GaussianBlur((smooth - mean) ** 2, (0, 0), CONTRAST_SIGMA * scale)

    return (smooth - mean) / np.sqrt(variance + 1e-3)


def local_orientation(normalized_image, scale=1.0):
    """Ridge tangent angle per pixel in [0, pi), from the structure tensor."""
    smooth = cv2.GaussianBlur(normalized_image, (0, 0), 1.0)
    grad_x = cv2.Sobel(smooth, cv2.CV_32F, 1, 0)
    grad_y = cv2.Sobel(smooth, cv2.CV_32F, 0, 1)
    sigma = ORIENTATION_SIGMA * scale
    j_xx = cv2.GaussianBlur(grad_x * grad_x, (0, 0), sigma)
    j_yy = cv2.GaussianBlur(grad_y * grad_y, (0, 0), sigma)
    j_xy = cv2.GaussianBlur(grad_x * grad_y, (0, 0), sigma)

    return (0.5 * np.arctan2(2 * j_xy, j_xx - j_yy) + np.pi / 2) % np.pi


def _gabor_pair(period):
    radius = int(np.ceil(3 * 0.45 * period))
    v, u = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(np.float32)
    envelope = np.exp(-0.5 * (u / (0.35 * period)) ** 2 - 0.5 * (v / (0.45 * period)) ** 2)
    real = envelope * np.cos(2 * np.pi * v / period)
    imag = envelope * np.sin(2 * np.pi * v / period)
    real -= envelope * real.sum() / envelope.sum()  # no response to a constant

    return real, imag


def sense_score(normalized_image, x, y, angle, scale=1.0):
    """Signed score in about [-1, 1] for a minutia at (x, y) with ridge tangent `angle`.

    Positive: `angle` already points in the conventional direction. Negative: it has to be
    turned by pi. The magnitude is a confidence.
    """
    sigma = WINDOW_SIGMA * scale
    half = int(np.ceil(3.6 * sigma))
    size = 2 * half + 1
    cos, sin = np.cos(angle), np.sin(angle)
    # patch rotated so that `angle` runs along +u (columns)
    transform = np.array([[cos, -sin, x - (cos - sin) * half],
                          [sin, cos, y - (sin + cos) * half]], dtype=np.float32)
    patch = cv2.warpAffine(normalized_image, transform, (size, size),
                           flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_REFLECT)

    v, u = np.mgrid[0:size - 1, 0:size - 1] - (half - 0.5)
    window = np.exp(-0.5 * (u ** 2 + v ** 2) / sigma ** 2)

    total = 0.0
    for period in RIDGE_PERIODS:
        real, imag = _gabor_pair(period * scale)
        phase = np.arctan2(
            -cv2.filter2D(patch, cv2.CV_32F, imag, borderType=cv2.BORDER_REFLECT),
            cv2.filter2D(patch, cv2.CV_32F, real, borderType=cv2.BORDER_REFLECT))
        d_v = _wrap(phase[1:, :] - phase[:-1, :])
        d_u = _wrap(phase[:, 1:] - phase[:, :-1])
        # phase winding around every pixel cell: +-1 at a ridge dislocation, 0 elsewhere
        charge = np.round((d_v[:, 1:] - d_u[1:, :] - d_v[:, :-1] + d_u[:-1, :]) / (2 * np.pi))
        total += float((charge * window).sum())

    return total / len(RIDGE_PERIODS)


def refine_minutiae_direction(image, minutiae, image_dpi=REFERENCE_DPI):
    """Set the direction sense of every minutia from the image.

    `minutiae` is an (N, >=3) array with x, y, angle in the first three columns. Only the
    angle modulo pi is used; when it differs by more than 25 degrees from the ridge
    orientation measured in the image, the measured orientation replaces it.

    Returns a copy of `minutiae` with the new angles in [0, 2 pi) and an array with one
    confidence per minutia (about 0-1; below about 0.15 the sense is a guess).
    """
    refined = np.array(minutiae, dtype=float, copy=True)
    confidence = np.zeros(refined.shape[0])

    if refined.size == 0:
        return refined, confidence

    scale = float(image_dpi) / REFERENCE_DPI
    normalized = normalize_contrast(image, scale)
    orientation = local_orientation(normalized, scale)
    rows, cols = orientation.shape

    for idx in range(refined.shape[0]):
        x, y = refined[idx, 0], refined[idx, 1]
        angle = refined[idx, 2] % np.pi
        row = min(max(int(round(y)), 0), rows - 1)
        col = min(max(int(round(x)), 0), cols - 1)
        measured = float(orientation[row, col])

        # distance between two orientations, both modulo pi
        if abs(_wrap(2 * (angle - measured))) / 2 > ORIENTATION_GUARD:
            angle = measured

        score = sense_score(normalized, float(x), float(y), angle, scale)

        refined[idx, 2] = (angle if score >= 0 else angle + np.pi) % (2 * np.pi)
        confidence[idx] = abs(score)

    return refined, confidence
