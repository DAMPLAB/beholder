'''
--------------------------------------------------------------------------------
Description:

Roadmap:

Written by W.R. Jackson <wrjackso@bu.edu>, DAMP Lab 2020
--------------------------------------------------------------------------------
'''
import copy

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


# ------------------------------------------------------------------------------
def plot_histogram_notebook(input_array: np.ndarray):
    hist, bins = np.histogram(input_array)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.savefig('hist_tshoot.png')
    # plt.show()


def plot_notebook(input_array: np.ndarray):
    plt.imshow(input_array, cmap='gray')
    plt.show()


# ------------------------------------------------------------------------------
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
    bin_limit = bins[2]
    for contour in contour_list:
        polygon_contour = cv2.approxPolyDP(contour, 3, True)
        bbox_list.append(cv2.boundingRect(polygon_contour))
    for i in range(len(contour_list)):
        # Going to be colorized in a subsequent call.
        c_stats = cell_stats[i]
        # if c_stats.sum_signal < bin_limit:
        #     print('Hewwo')
        # else:
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


def plot_total(
        total_statistics: List[Tuple[float, float, float]],
        context: str = 'save',
):
    '''

    Args:
        total_statistics:

    Returns:

    '''
    channel_one_stats = []
    channel_two_stats = []
    out_list = []
    fig = plt.figure()
    canvas = FigureCanvasAgg(fig)
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
    ax1.fill_between(
        time_scale,
        channel_one_pos,
        channel_one_neg,
        alpha=.5,
        color='green',
    )
    ax1.fill_between(
        time_scale,
        channel_two_pos,
        channel_two_neg,
        alpha=.5,
        color='red',
    )
    ax1.plot(time_scale[::-1], channel_one_median[::-1], color='green')  #
    ax1.plot(time_scale[::-1], channel_two_median[::-1], color='red')  #
    if context == 'save':
        plt.savefig('example.png')
    if context == 'subplot':
        canvas.draw()
        buf = canvas.buffer_rgba()
        buf_1 = buf[:]
        out_list.append(buf_1)
    # plt.clf()
    # plt.cla()
    # plt.close()
    ax2.plot(time_scale[::-1], channel_one_cell_count[::-1], color='blue')  #
    ax2.plot(time_scale[::-1], channel_two_cell_count[::-1], color='purple')  #
    if context == 'save':
        plt.savefig('example1.png')
    if context == 'subplot':
        canvas.draw()
        buf = canvas.buffer_rgba()
        buf_2 = buf[:]
        out_list.append(buf_2)
    # plt.legend()
    # plt.show()
    if context == 'subplot':
        canvas.draw()
        buf = canvas.buffer_rgba()
        return buf


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


def stat_splitter(
        total_statistics: List[Tuple[float, float]],
):
    channel_one_stats = []
    channel_two_stats = []
    for stat_pair in total_statistics:
        c1_stat, c2_stat = stat_pair
        channel_one_stats.append(c1_stat)
        channel_two_stats.append(c2_stat)
    return channel_one_stats, channel_two_stats


def plot_cellular_signal(channel_one_stats, channel_two_stats):
    fig = plt.figure()
    canvas = FigureCanvasAgg(fig)
    plt.title('Cellular Signal')
    channel_one_median = [stat[0] for stat in channel_one_stats]
    channel_one_std_dev = [stat[1] for stat in channel_one_stats]
    channel_two_median = [stat[0] for stat in channel_two_stats]
    channel_two_std_dev = [stat[1] for stat in channel_two_stats]
    channel_one_pos = np.add(channel_one_median, channel_one_std_dev)
    channel_one_neg = np.subtract(channel_one_median, channel_one_std_dev)
    channel_two_pos = np.add(channel_two_median, channel_two_std_dev)
    channel_two_neg = np.subtract(channel_two_median, channel_two_std_dev)
    channel_one_neg = np.where(channel_one_neg < 0, 0, channel_one_neg)
    channel_two_neg = np.where(channel_two_neg < 0, 0, channel_two_neg)
    time_scale = range(len(channel_one_stats))
    # We want the lower band, the higher band, and the actual value.
    plt.fill_between(
        time_scale[::-1],
        channel_one_pos[::-1],
        channel_one_neg,
        alpha=.5,
        color='green',
    )
    plt.fill_between(
        time_scale[::-1],
        channel_two_pos[::-1],
        channel_two_neg[::-1],
        alpha=.5,
        color='red',
    )
    plt.plot(time_scale[::-1], channel_one_median[::-1], color='green')  #
    plt.plot(time_scale[::-1], channel_two_median[::-1], color='red')  #
    canvas.draw()
    buf = canvas.buffer_rgba()
    return buf


def plot_cell_count(channel_one_stats, channel_two_stats):
    fig = plt.figure()
    canvas = FigureCanvasAgg(fig)
    plt.title('Cell Count')
    channel_one_cell_count = [stat[2] for stat in channel_one_stats]
    channel_two_cell_count = [stat[2] for stat in channel_two_stats]
    time_scale = range(len(channel_one_stats))
    # We want the lower band, the higher band, and the actual value.
    plt.plot(time_scale[::-1], channel_one_cell_count[::-1], color='red')  #
    plt.plot(time_scale[::-1], channel_two_cell_count[::-1], color='green')  #
    canvas.draw()
    buf = canvas.buffer_rgba()
    return buf


def plot_double_histogram(processed_frame, raw_frame, axis, plot_name):
    '''

    Returns:

    '''
    fig = plt.figure()
    canvas = FigureCanvasAgg(fig)
    plt.title(plot_name)
    p_hist, p_bins = np.histogram(processed_frame, bins=100)
    r_hist, r_bins = np.histogram(raw_frame, bins=100)
    center = (p_bins[:-1] + p_bins[1:]) / 2
    axis.bar(center, p_hist, align='center', width=100, color='green')
    axis.bar(center, r_hist, align='center', width=100, color='red')
    axis.set_xlim([0, 100])
    axis.set_ylim([0, 1000000])
    axis.set_xticklabels([])
    axis.set_yticklabels([])
    canvas.draw()
    buf = canvas.buffer_rgba()
    return buf


def generate_image_canvas(
        processed_frame: np.ndarray,
        raw_frame: np.ndarray,
        stats_list: List,
        title: str,
        stats_final_size: int,
):
    fig = plt.figure(figsize=(13, 6))
    canvas = FigureCanvasAgg(fig)
    grid = plt.GridSpec(7, 6, hspace=0.0, wspace=0.0)
    plt.title(f'{title}')
    # Padding the Array to our final point
    stats_list += [[(0, 0, 0), (0, 0, 0)]] * (stats_final_size - len(stats_list))
    axis_1 = fig.add_subplot(grid[:6, :5])
    axis_2 = fig.add_subplot(grid[0:3, 4:6])
    axis_3 = fig.add_subplot(grid[3:6, 4:6])
    axis_4 = fig.add_subplot(grid[6:7, :])
    # axis_5 = fig.add_subplot(grid[6:7, 3:6])
    c1_stats, c2_stats = stat_splitter(stats_list)
    cell_signal = plot_cellular_signal(c1_stats, c2_stats)
    cell_count = plot_cell_count(c1_stats, c2_stats)
    # TODO: Work via Mutation.
    plot_double_histogram(processed_frame, raw_frame, axis_4, 'Processed Frame')
    plot_histogram_notebook(raw_frame)
    array_list = [processed_frame, cell_signal, cell_count]
    axis_list = [axis_1, axis_2, axis_3]
    for array, axis in zip(array_list, axis_list):
        axis.imshow(array, interpolation='nearest')
        plt.axis('off')
        axis.set_xticklabels([])
        axis.set_yticklabels([])
        axis.set_aspect('equal')
    plt.tight_layout()
    plt.text(0.1, 0.9, 'matplotlib', ha='center', va='center')
    canvas.draw()
    buf = canvas.buffer_rgba()
    return np.asarray(buf)
