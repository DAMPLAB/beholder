'''
--------------------------------------------------------------------------------
Description:

Roadmap:

Written by W.R. Jackson <wrjackso@bu.edu>, DAMP Lab 2020
--------------------------------------------------------------------------------
'''
import numpy as np
from signal_processing import (
    signal_transform,
    sigpro_utility,
    graphing,
    stats,
)


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
    sigpro_utility.display_frame(initial_frame, 'initial')
    out_frame = signal_transform.apply_brightness_contrast(initial_frame)
    # out_frame = signal_transform.percentile_threshold(initial_frame)
    # sigpro_utility.display_frame(out_frame, 'contrast')
    # out_frame = signal_transform.invert_image(out_frame)
    sigpro_utility.display_frame(out_frame, 'invert')
    out_frame = signal_transform.remove_background(out_frame)
    sigpro_utility.display_frame(out_frame, 'background removal')
    out_frame = signal_transform.downsample_image(out_frame)
    sigpro_utility.display_frame(out_frame, 'downsample')
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
    #  confounded by
    filtered_contours = signal_transform.cellular_highpass_filter(contours)
    return filtered_contours


def generate_mask(input_frame: np.ndarray, contours):
    out_frame = graphing.draw_contours(
        input_frame,
        contours,
        colouration='rainbow',
    )
    return out_frame


def segmentation_pipeline(input_fn: str):
    grey_frame, red_frame, green_frame = sigpro_utility.open_microscopy_image(
        input_fn
    )
    c_red_frame = preprocess_color_channel(red_frame, 'red')
    # sigpro_utility.display_frame(green_frame)
    c_green_frame = preprocess_color_channel(green_frame, 'green')
    # sigpro_utility.display_frame(green_frame)
    mask_frame = np.zeros_like(grey_frame)
    contours = preprocess_initial_grey_and_find_contours(grey_frame)
    g_contours = preprocess_initial_color_and_find_contours(green_frame)
    # g_contours = contour_filtration(g_contours)
    contours = contour_filtration(contours)
    green_cell_signals = stats.fluoresence_detection(
        grey_frame,
        green_frame,
        contours,
    )
    labeled_green = graphing.label_cells(
        signal_transform.downsample_image(green_frame),
        contours,
        green_cell_signals,
    )
    red_cell_signals = stats.fluoresence_detection(
        grey_frame,
        red_frame,
        contours,
    )
    labeled_green = graphing.label_cells(
        signal_transform.downsample_image(green_frame),
        contours,
        green_cell_signals,
    )
    labeled_green = signal_transform.colorize_frame(labeled_green, 'green')
    d_grey_frame = signal_transform.downsample_image(grey_frame)
    out_frame = signal_transform.combine_frame(
        d_grey_frame,
        c_green_frame,
    )
    out_frame = signal_transform.combine_frame(
        out_frame,
        labeled_green,
    )

    mask_frame = generate_mask(mask_frame, contours)
    sigpro_utility.display_frame(mask_frame)


if __name__ == '__main__':
    input_file = "../data/agarose_pads/SR15_1mM_IPTG_Agarose_TS_1h_1.nd2"
    segmentation_pipeline(input_file)
