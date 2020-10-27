'''
--------------------------------------------------------------------------------
Description:
Wrappers for various array transformation utilities commonly used in cellular
segmentation

Roadmap:

Written by W.R. Jackson <wrjackso@bu.edu>, DAMP Lab 2020
--------------------------------------------------------------------------------
'''
import random as rng
from typing import (
    List,
    Tuple,
)

import cv2
import numpy as np


# ------------------------------ Array Transforms ------------------------------
def downsample_image(input_frame: np.ndarray) -> np.ndarray:
    '''
    Downsamples a 16bit input image to an 8bit image. Will result in loss
    of signal fidelity due to loss of precision.

    Args:
        input_frame: Input numpy array

    Returns:
        Downsampled ndarray
    '''
    return (input_frame / 256).astype('uint8')


def normalize_frame(input_frame: np.ndarray) -> np.ndarray:
    '''

    Args:
        input_frame:

    Returns:

    '''
    return (255 * input_frame / np.max(input_frame)).astype(np.uint8)


def percentile_threshold(input_frame, low_th=2, high_th=98) -> np.ndarray:
    '''

    Args:
        input_frame:
        low_th:
        high_th:

    Returns:

    '''
    low_threshold = np.percentile(input_frame, low_th)
    high_threshold = np.percentile(input_frame, high_th)
    ret, thresh = cv2.threshold(input_frame, low_threshold, high_threshold, 0)
    return thresh


def invert_image(input_frame: np.ndarray) -> np.ndarray:
    '''

    Args:
        input_frame:

    Returns:

    '''
    return np.invert(input_frame)


def remove_background(
        input_frame: np.ndarray,
        adjustment: float = 1.0,
) -> np.ndarray:
    '''

    Args:
        input_frame:
        adjustment:

    Returns:

    '''
    background_level = np.mean(input_frame) * adjustment
    return input_frame - background_level


def kernel_smoothing(input_frame: np.ndarray, k_size: int = 9) -> np.ndarray:
    '''

    Args:
        input_frame:
        k_size:

    Returns:

    '''
    kernel = np.array([[-1, -1, -1], [-1, k_size, -1], [-1, -1, -1]])
    return cv2.filter2D(input_frame, -1, kernel)


def unsharp_mask(
        input_frame: np.ndarray,
        kernel_size: Tuple[int, int] = (5, 5),
        sigma: float = 1.0,
        amount: float = 1.0,
        threshold: float = 0.0
) -> np.ndarray:
    '''
    Return a sharpened version of the image, using an unsharp mask.

    Args:
        input_frame:
        kernel_size:
        sigma:
        amount:
        threshold:

    Returns:

    '''
    blurred = cv2.GaussianBlur(input_frame, kernel_size, sigma)
    sharpened = float(amount + 1) * input_frame - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(input_frame - blurred) < threshold
        np.copyto(sharpened, input_frame, where=low_contrast_mask)
    return sharpened


# I think that np.ndarray is wrong. Should be like a list of list of floats.
def find_contours(input_frame: np.ndarray) -> np.ndarray:
    '''

    Args:
        input_frame:

    Returns:

    '''
    contours, hierarchy = cv2.findContours(
        input_frame,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE,
    )
    return contours


def draw_convex_hull(
        edges: List[List[float]],
        contours: List[List[float]],
) -> np.ndarray:
    '''
    Find the convex hull object for each contour

    Args:
        edges:
        contours:

    Returns:

    '''
    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_list.append(hull)
    # Draw contours + hull results
    drawing = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        cv2.drawContours(drawing, contours, i, color)
        cv2.drawContours(drawing, hull_list, i, color)
    return drawing


def gaussian_blur(input_frame: np.ndarray, kernel: int = 3) -> np.ndarray:
    '''

    Args:
        input_frame:
        kernel:

    Returns:

    '''
    return cv2.GaussianBlur(input_frame, (kernel, kernel), 0)


def laplacian_operator(input_frame: np.ndarray) -> np.ndarray:
    '''

    Args:
        input_frame:

    Returns:

    '''
    return cv2.Laplacian(input_frame, cv2.CV_64F, ksize=3)


def otsu_thresholding(input_frame: np.ndarray) -> np.ndarray:
    '''

    Args:
        input_frame:

    Returns:

    '''
    blur = cv2.GaussianBlur(input_frame, (3, 3), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3


def auto_canny(input_frame: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    '''

    Args:
        input_frame:
        sigma:

    Returns:

    '''
    # compute the median of the single channel pixel intensities
    v = np.median(input_frame)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(input_frame, lower, upper)


def pyramid_mean_shift(
        input_frame: np.ndarray,
        lower: int = 21,
        upper: int = 51,
):
    '''

    Args:
        input_frame:
        lower:
        upper:

    Returns:

    '''
    return cv2.pyrMeanShiftFiltering(input_frame, lower, upper)


def colorize_frame(input_frame: np.ndarray, color: str) -> np.ndarray:
    '''
    Converts a 1-Channel Grayscale Image to RGB Space and then converts the
    prior image to a singular color.

    Args:
        input_frame: Input numpy array
        color: (red|blue|green) What color to convert to:

    Returns:
        Colorized ndarray

    '''
    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_GRAY2RGB)
    if color == 'green':
        input_frame[:, :, (0, 2)] = 0
    if color == 'red':
        input_frame[:, :, (1, 2)] = 0
    if color == 'blue':
        input_frame[:, :, (0, 1)] = 0
    return input_frame


def modify_contrast(
        input_frame: np.ndarray,
        alpha: int = 5,
        gamma: int = 127,
):
    '''
    Modifies the contrast of the input image. This is done through modifying
    the alpha and gamma. Alpha multiplies the underlying intensity while the
    gamma is a straight increase to the intensity.

    Args:
        input_frame: Input numpy array
        alpha: Alpha Value
        gamma: Gamma Value

    Returns:
        Contrast Modified ndarray

    '''
    return cv2.addWeighted(
        input_frame,
        alpha,
        input_frame,
        0,
        gamma,
    )


def apply_brightness_contrast(
        input_frame: np.ndarray,
        alpha=12.0,
        beta=0,
):
    '''
    Modifies the contrast of the input image. This is done through modifying
    the alpha and gamma. Alpha multiplies the underlying intensity while the
    gamma is a straight increase to the intensity. Basically, alpha is contrast
    and beta is brightness

    Args:
        input_frame: Input numpy array
        alpha: Alpha Value
        beta: Beta Value

    Returns:
        Contrast Modified ndarray

    '''
    return cv2.convertScaleAbs(
        input_frame,
        alpha=alpha,
        beta=beta,
    )


def empty_frame_check(input_frame: np.ndarray) -> bool:
    '''
    Checks to see if the inputted frame is empty.

    An implementation detail from the underlying nd2reader library we're using:
    Discontinuous or dropped frames are represented by NaN values instead of
    zeroes which will later be implicitly cast to zeroes when being persisted
    to disk. There does seem to be some places where there is a mismatch
    between channels which results in the resultant image being off in terms
    of stride.

    Args:
        input_frame: Input numpy array

    Returns:
        Whether or not the frame is 'emtpy'

    '''
    nan_converted_frame = np.nan_to_num(input_frame)
    if np.sum(nan_converted_frame) == 0:
        return True
    return False


def combine_frame(
        input_frame_one: np.ndarray,
        input_frame_two: np.ndarray,
        alpha: int = 1,
        beta: float = 0.75,
        gamma: float = 0,
) -> np.ndarray:
    '''

    Args:
        input_frame_one:
        input_frame_two:
        alpha:
        beta:
        gamma:

    Returns:

    '''
    # I assume we're never going to create greyscale on greyscale stacks
    frame_list = [input_frame_one, input_frame_two]
    for idx, frame in enumerate(frame_list):
        if len(frame.shape) < 3:
            frame_list[idx] = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(
        frame_list[0],
        alpha,
        frame_list[1],
        beta,
        gamma,
    )


# ----------------------------------- Filters ----------------------------------
def cellular_highpass_filter(contours):
    '''

    Args:
        contours:

    Returns:

    '''
    contour_areas = list(map(cv2.contourArea, contours))
    hist, bin_edges = np.histogram(contour_areas)
    out_list = []
    for c_value, c_area in zip(contours, contour_areas):
        percentile_bin = bin_edges[1]
        # TODO: This hardcoded value is due to large congregations of cells
        #   causing the bins to skew hella right tailed. Should be able to switch
        #   back to histogramic filtering once that's all figured out.
        if c_area > 100:
            out_list.append(c_value)
    return out_list
