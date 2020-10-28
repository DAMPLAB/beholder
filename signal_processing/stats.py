'''
--------------------------------------------------------------------------------
Description:

Roadmap:

Written by W.R. Jackson <wrjackso@bu.edu>, DAMP Lab 2020
--------------------------------------------------------------------------------
'''
import csv
import os
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
import pandas as pd


@dataclass
class CellSignal:
    contour_index: int
    sum_signal: float
    fluorescent_pixels: List[float]


def fluorescence_detection(
        grayscale_frame: np.ndarray,
        fluorescent_frame: np.ndarray,
        contour_list: List[np.ndarray],
):
    '''

    Args:
        grayscale_frame:
        fluorescent_frame:
        contour_list:

    Returns:

    '''
    # Cell Locations is a list containing an identifying contour index (so we
    # can map it to cells), the summation of the signal, and then every single
    # fluorescent indice if we feel like doing anything.
    cell_signals = []
    for contour_idx in range(len(contour_list)):
        contour_mask = np.zeros_like(grayscale_frame)
        cv2.drawContours(
            contour_mask,
            contour_list,
            contour_idx,
            color=255,
            thickness=-1
        )
        mask_indices = np.where(contour_mask == 255)
        fluorescent_intensities = fluorescent_frame[mask_indices[0], mask_indices[1]]
        # Noise floor. 256 >> 2 = 32?
        if np.mean(fluorescent_intensities) < 31:
            pass
        else:
            new_cell = CellSignal(
                contour_index=contour_idx,
                sum_signal=np.sum(fluorescent_intensities),
                fluorescent_pixels=fluorescent_intensities,
            )
            cell_signals.append(new_cell)
    return cell_signals


def generate_frame_stats(
        c1_cell_stats: List[CellSignal],
        c2_cell_stats: List[CellSignal],
):
    '''

    Args:
        c1_cell_stats:
        c2_cell_stats:

    Returns:

    '''
    out_list = []
    for cell_set in [c1_cell_stats, c2_cell_stats]:
        c_signal = [cell_st.sum_signal for cell_st in cell_set]
        hist, bins = np.histogram(c_signal)
        bin_limit = bins[3]
        filtered_set = list(filter(lambda x: x.sum_signal < bin_limit, cell_set))
        f_signal = [fil_st.sum_signal for fil_st in filtered_set]
        median_signal = np.median(f_signal)
        std_dev = np.std(f_signal)
        out_list.append((median_signal, std_dev, len(filtered_set)))
    return out_list


def write_stat_record(
        total_statistics,
        record_name: str,
):
    channel_one_stats = []
    channel_two_stats = []
    for index, stat_pair in enumerate(total_statistics):
        c1_stat, c2_stat = stat_pair
        channel_one_stats.append(c1_stat)
        channel_two_stats.append(c2_stat)
    dt = {
        'index': list(range(len(total_statistics))),
        'ch_1_fluorescence': [stat[0] for stat in channel_one_stats],
        'ch_1_std_dev': [stat[1] for stat in channel_one_stats],
        'ch_1_cell_count': [stat[2] for stat in channel_one_stats],
        'ch_2_fluorescence': [stat[0] for stat in channel_two_stats],
        'ch_2_std_dev': [stat[1] for stat in channel_two_stats],
        'ch_2_cell_count': [stat[2] for stat in channel_two_stats],
    }
    # TODO: Sloppy, time crunch.
    df = pd.DataFrame.from_dict(dt)
    df.to_csv(record_name)
    print(os.getcwd())
    # try:
    #     with open(record_name, 'w') as csvfile:
    #         writer = csv.DictWriter(csvfile, fieldnames=list(dt.keys()))
    #         writer.writeheader()
    #         for data in dt:
    #             writer.writerow(data)
    # except IOError as e:
    #     print(f'I/O Error: {e}')
    #