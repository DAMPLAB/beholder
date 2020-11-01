'''
--------------------------------------------------------------------------------
Description:

Roadmap:

Written by W.R. Jackson <wrjackso@bu.edu>, DAMP Lab 2020
--------------------------------------------------------------------------------
'''
from fractions import Fraction
import random as rng
from typing import (
    List,
    Tuple,
)

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.figure as m_figure
from signal_processing.stats import CellSignal
from PIL import Image, ImageDraw, ImageFont
from matplotlib.backends.backend_agg import FigureCanvasAgg


def plot_histogram(input_array: np.ndarray):
    hist, bins = np.histogram(input_array)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()


def draw_contours(
        input_frame: np.ndarray,
        contour_list: List[np.ndarray],
        stroke: int = 3,
) -> np.ndarray:
    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_GRAY2RGB)
    for contour_idx in range(len(contour_list)):
        input_frame = cv2.drawContours(
            input_frame,
            contour_list,
            contour_idx,
            0,
            stroke,
        )
    return input_frame


def draw_mask(
        input_frame: np.ndarray,
        contour_list: List[np.ndarray],
        stroke: int = -1,
        colouration: str = 'white',
) -> np.ndarray:
    if colouration not in ['white', 'rainbow']:
        print('Current colouration not supported.')
    if colouration == 'white':
        for contour_idx in range(len(contour_list)):
            input_frame = cv2.drawContours(
                input_frame,
                contour_list,
                contour_idx,
                255,
                stroke,
            )
        return input_frame

    if colouration == 'rainbow':
        # TODO: I'm assuming that it's grayscale here. I should make sure to
        #  make this detect our current colorscale.
        rgb_frame = np.zeros(
            (input_frame.shape[0], input_frame.shape[1], 3),
            dtype=np.uint8,
        )
        for contour_idx in range(len(contour_list)):
            color = (
                rng.randint(0, 256),
                rng.randint(0, 256),
                rng.randint(0, 256)
            )
            rgb_frame = cv2.drawContours(
                rgb_frame,
                contour_list,
                contour_idx,
                color,
                stroke,
            )
        return rgb_frame


def write_multiline(
        input_frame: np.ndarray,
        input_str: str,
        loc_x: int,
        loc_y: int,
):
    write_frame = np.zeros(
        (input_frame.shape[0], input_frame.shape[1]),
        dtype=np.uint8,
    )
    cv2.putText(
        write_frame,
        input_str,
        (loc_x, loc_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        255,
        2,
        cv2.LINE_AA,
    )
    output_frame = cv2.addWeighted(
        input_frame,
        1,
        write_frame,
        0.5,
        0,
    )
    return output_frame


def label_cells(
        input_frame: np.ndarray,
        contour_list: List[np.ndarray],
        cell_stats: List[CellSignal],
):
    # Get Bounding Box for Contour
    # Outline Bounding Box in very thin line
    # Next to bounding box put
    # input_frame = np.ndarray(input_frame)
    bbox_list = []
    c_signal = [cell_st.sum_signal for cell_st in cell_stats]
    hist, bins = np.histogram(c_signal, bins=100)
    bin_limit = bins[8]
    for contour in contour_list:
        polygon_contour = cv2.approxPolyDP(contour, 3, True)
        bbox_list.append(cv2.boundingRect(polygon_contour))
    for i in range(len(contour_list)):
        # Going to be colorized in a subsequent call.
        c_stats = cell_stats[i]
        if c_stats.sum_signal < bin_limit:
            continue
        else:
            input_frame = cv2.rectangle(
                input_frame,
                (bbox_list[i][0], bbox_list[i][1]),
                ((bbox_list[i][0] + bbox_list[i][2]),
                 bbox_list[i][1] + bbox_list[i][3]),
                255,
                1)
            input_frame = write_multiline(
                input_frame,
                f'{c_stats.sum_signal}',
                bbox_list[i][0],
                bbox_list[i][1],
            )
    return input_frame


def plot_total(total_statistics: List[Tuple[float, float]]):
    '''

    Args:
        total_statistics:

    Returns:

    '''
    channel_one_stats = []
    channel_two_stats = []
    for stat_pair in total_statistics:
        c1_stat, c2_stat = stat_pair
        channel_one_stats.append(c1_stat)
        channel_two_stats.append(c2_stat)
    channel_one_median = [stat[0] for stat in channel_one_stats]
    channel_one_std_dev = [stat[1] for stat in channel_one_stats]
    channel_one_cell_count = [stat[2] for stat in channel_one_stats]
    channel_two_median = [stat[0] for stat in channel_two_stats]
    channel_two_std_dev = [stat[1] for stat in channel_two_stats]
    channel_two_cell_count = [stat[2] for stat in channel_two_stats]
    channel_one_pos = np.add(channel_one_median, channel_one_std_dev)
    channel_one_neg = np.subtract(channel_one_median, channel_one_std_dev)
    channel_two_pos = np.add(channel_two_median, channel_two_std_dev)
    channel_two_neg = np.subtract(channel_two_median, channel_two_std_dev)
    time_scale = range(len(channel_one_stats))
    # We want the lower band, the higher band, and the actual value.

    plt.fill_between(
        time_scale,
        channel_one_pos,
        channel_one_neg,
        alpha=.5,
        color='green',
    )
    plt.fill_between(
        time_scale,
        channel_two_pos,
        channel_two_neg,
        alpha=.5,
        color='red',
    )
    plt.plot(time_scale, channel_one_median, color='green')
    plt.plot(time_scale, channel_two_median, color='red')
    plt.savefig('example.png')
    plt.figure(2)
    plt.plot(time_scale, channel_one_cell_count, color='lime')
    plt.plot(time_scale, channel_two_cell_count, color='lightpink')
    plt.savefig('example1.png')

    plt.legend()
    # plt.show()


def generate_multiplot(
        frame_list: List[np.ndarray],
        frame_annotations: List[str],
        context: str = 'active_frame',
):
    fig = plt.figure(figsize=(16, 3))
    canvas = FigureCanvasAgg(fig)
    gs1 = gridspec.GridSpec(1, 4)
    gs1.update(wspace=0.2, hspace=0.02)
    plt.margins(.5, .5)
    for i, anno in enumerate(frame_annotations):
        ax1 = plt.subplot(gs1[i])
        active_frame = frame_list[i]
        if not i:
            ax1.imshow(active_frame[
                       int(active_frame.shape[0] / 2):int(active_frame.shape[0]),
                       int(active_frame.shape[1] / 2):int(active_frame.shape[1])
                       ],
                       cmap='gray')
        else:
            ax1.imshow(active_frame[
                       int(active_frame.shape[0] / 2):int(active_frame.shape[0]),
                       int(active_frame.shape[1] / 2):int(active_frame.shape[1])
                       ],
                       interpolation='nearest')
        ax1.set_title(anno)
        plt.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
        autoAxis = ax1.axis()
        rec = plt.Rectangle((autoAxis[0] - 0.7, autoAxis[2] - 0.2),
                            (autoAxis[1] - autoAxis[0]) + 1,
                            (autoAxis[3] - autoAxis[2]) + 0.4, fill=False, lw=2)
        rec = ax1.add_patch(rec)
        rec.set_clip_on(False)
    plt.axis('off')
    if context == 'to_disk':
        plt.savefig(
            "test1.jpg",
            bbox_inches='tight',
            pad_inches=.2,
        )
    if context == 'active_frame':
        canvas.draw()
        buf = canvas.buffer_rgba()
        return np.asarray(buf)
