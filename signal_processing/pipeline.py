'''
--------------------------------------------------------------------------------
Description:

Roadmap:

Written by W.R. Jackson <wrjackso@bu.edu>, DAMP Lab 2020
--------------------------------------------------------------------------------
'''
import datetime
import multiprocessing as mp
from typing import Tuple

import imageio
import numpy as np
import tqdm
from pygifsicle import optimize
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

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


def generate_mask(input_frame: np.ndarray, contours):
    out_frame = graphing.draw_mask(
        input_frame,
        contours,
        colouration='rainbow',
    )
    return out_frame


def segmentation_pipeline(
        input_frames: Tuple[np.ndarray, np.ndarray, np.ndarray],
):
    grey_frame, red_frame, green_frame = input_frames
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
    frame_stats = stats.generate_frame_stats(
        green_cell_signals,
        red_cell_signals,
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
    mask_frame = generate_mask(mask_frame, contours)
    return out_frame, frame_stats, mask_frame


if __name__ == '__main__':
    # We need a function that takes an ND2, extracts all of the color channels,
    # and returns tuples of each frame with it's constiuent channels as a big
    # ass list
    fn = "../data/New_SR_1_5_MC_TS10h.nd2"
    frames = sigpro_utility.parse_nd2_file(fn)
    final_frame = []
    final_stats = []
    final_mask = []
    for frame in tqdm.tqdm(frames):
        out_frame, frame_stats, mask_frame = segmentation_pipeline(frame)
        final_frame.append(out_frame)
        final_stats.append(frame_stats)
    graphing.plot_total(final_stats)
    stats.write_stat_record(
        final_stats,
        f'{(fn.split("/")[-1])[:-3]}_{datetime.datetime.now().date()}.csv'
    )
    imageio.mimsave(f'test1.gif', final_frame)
    imageio.mimsave(f'mask.gif', final_mask)
    optimize('test1.gif')
    optimize('mask.gif')
