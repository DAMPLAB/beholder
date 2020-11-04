'''
--------------------------------------------------------------------------------
Description:

Roadmap:

Written by W.R. Jackson <wrjackso@bu.edu>, DAMP Lab 2020
--------------------------------------------------------------------------------
'''
import copy
from typing import (
    Tuple,
)

import numpy as np
from signal_processing import (
    signal_transform,
    sigpro_utility,
    graphing,
    stats,
)


def calculate_attic(
        fp: str,
        swatch_point_0: Tuple[int, int],
        swatch_point_1: Tuple[int, int],
        mask_point_0: Tuple[int, int],
        mask_point_1: Tuple[int, int],
):
    '''

    Args:
        fp:
        swatch_point_0:
        swatch_point_1:
        mask_point_0:
        mask_point_1:

    Returns:

    '''
    reference_frame = sigpro_utility.get_initial_image_nd2(fp)
    swatch_median = np.median(signal_transform.crop_from_points(
        reference_frame,
        swatch_point_0,
        swatch_point_1,
    )
    )
    mask_median = np.median(signal_transform.crop_from_points(
        reference_frame,
        mask_point_0,
        mask_point_1,
    )
    )
    return abs(swatch_median + mask_median)



def preprocess_initial_grey_and_find_contours(initial_frame: np.ndarray):
    # Each image transform should be giving us back an np.ndarray of the same
    # shape and size.
    out_frame = signal_transform.percentile_threshold(initial_frame)
    out_frame = signal_transform.invert_image(out_frame)
    out_frame = signal_transform.remove_background(out_frame)
    out_frame = signal_transform.downsample_image(out_frame)
    contours = signal_transform.find_contours(out_frame)
    return contours


def preprocess_initial_color_and_find_contours(initial_frame: np.ndarray):
    # Each image transform should be giving us back an np.ndarray of the same
    # shape and size.
    out_frame = signal_transform.downsample_image(initial_frame)
    out_frame = signal_transform.apply_brightness_contrast(
        out_frame,
        alpha=2,
        beta=0,
    )
    out_frame = signal_transform.percentile_threshold(out_frame, 80, 98)
    out_frame = signal_transform.invert_image(out_frame)
    out_frame = signal_transform.remove_background(out_frame)
    out_frame = signal_transform.downsample_image(out_frame)
    contours = signal_transform.find_contours(out_frame)
    return contours


def preprocess_color_channel(
        initial_frame: np.ndarray,
        color: str,
        alpha: float = 12,
        beta: int = 0,
):
    out_frame = signal_transform.downsample_image(initial_frame)
    out_frame = signal_transform.apply_brightness_contrast(
        out_frame,
        alpha,
        beta,
    )
    out_frame = signal_transform.colorize_frame(out_frame, color)
    return out_frame


def contour_filtration(contours):
    # TODO: We then need to refine our approach in terms of segmentation either
    #  via eroding or some other mechanism. I think edge delineation is being
    #  confounded by the lack of depth in the microscopy image and the
    #  microfluidic device it's being housed in.
    filtered_contours = signal_transform.cellular_highpass_filter(contours)
    return filtered_contours


def generate_contours(input_frame: np.ndarray, contours):
    out_frame = graphing.draw_contours(
        input_frame,
        contours,
    )
    return out_frame


def generate_mask(input_frame: np.ndarray, contours):
    out_frame = graphing.draw_mask(
        input_frame,
        contours,
        colouration='rainbow',
    )
    return out_frame


def segmentation_pipeline(
        input_fn: str,
        mask_p0: Tuple[int, int],
        mask_p1: Tuple[int, int],
        normalized_mask: int,
):
    grey_frame, red_frame, green_frame = sigpro_utility.open_microscopy_image(
        input_fn
    )
    grey_frame = signal_transform.mask_subselection(
        grey_frame,
        mask_p0,
        mask_p1,
        normalized_mask,
    )
    prepro_frame = copy.copy(grey_frame)
    c_red_frame = preprocess_color_channel(red_frame, 'red')
    c_green_frame = preprocess_color_channel(green_frame, 'green')
    mask_frame = np.zeros_like(grey_frame)
    contours = preprocess_initial_grey_and_find_contours(grey_frame)
    contours = contour_filtration(contours)
    green_cell_signals = stats.fluorescence_detection(
        grey_frame,
        green_frame,
        contours,
    )
    red_cell_signals = stats.fluorescence_detection(
        grey_frame,
        red_frame,
        contours,
    )
    labeled_green = graphing.label_cells(
        signal_transform.downsample_image(green_frame),
        contours,
        green_cell_signals,
    )
    labeled_red = graphing.label_cells(
        signal_transform.downsample_image(red_frame),
        contours,
        red_cell_signals,
    )
    labeled_green = signal_transform.colorize_frame(labeled_green, 'green')
    labeled_red = signal_transform.colorize_frame(labeled_red, 'red')
    d_grey_frame = signal_transform.downsample_image(grey_frame)
    out_frame = signal_transform.combine_frame(
        d_grey_frame,
        c_red_frame,
    )
    out_frame = signal_transform.combine_frame(
        out_frame,
        labeled_red,
    )
    out_frame = signal_transform.combine_frame(
        out_frame,
        c_green_frame,
    )
    out_frame = signal_transform.combine_frame(
        out_frame,
        labeled_green,
    )
    contour_frame = copy.copy(prepro_frame)
    contour_frame = generate_contours(contour_frame, contours)
    mask_frame = generate_mask(mask_frame, contours)
    frame_list = [grey_frame, contour_frame, mask_frame, out_frame]
    frame_annotations = ['Preprocess', 'Contour', 'Mask', 'Output']
    graphing.generate_multiplot(frame_list, frame_annotations, 'to_disk')


if __name__ == '__main__':
    swatch_point_1 = (657, 209)
    swatch_point_2 = (817, 696)
    mask_point_1 = (967, 0)
    mask_point_2 = (1288, 1039)
    input_file = "../data/agarose_pads/SR15_1mM_IPTG_Agarose_TS_1h_1.nd2"
    normalize_value = calculate_attic(
        input_file,
        swatch_point_1,
        swatch_point_2,
        mask_point_1,
        mask_point_2,
    )
    segmentation_pipeline(
        input_file,
        mask_point_1,
        mask_point_2,
        normalize_value,
    )
