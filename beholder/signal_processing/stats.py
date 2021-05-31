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

from beholder.signal_processing.signal_transform import (
    debug_image,
)
from beholder.ds import (
    TiffPackage,
)


def debug_visualization(input_frame: np.ndarray, name: str):
    print(f'Debug for {name}')
    print(f'Total Value for Frames {np.sum(input_frame)}')
    print(f'Shape of Array {input_frame.shape}')
    debug_image(input_frame, name)
    print('------')


@dataclass
class CellSignal:
    contour_index: int
    sum_signal: float
    fluorescent_pixels: List[float]


@dataclass
class CellStats:
    raw_signal: List[float]
    fl_signal: List[float]
    median_signal: float
    std_dev: float
    filtered_len: int

def calculate_raw_fluorescence(input_frame: np.ndarray) -> int:
    return np.sum(input_frame)



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
    # fluorescent_frame = downsample_image(fluorescent_frame)

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
        # print(np.mean(fluorescent_intensities))
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


def fluorescence_filtration(
        grayscale_frame: np.ndarray,
        fluorescent_frame: np.ndarray,
        primary_contour_list: List[np.ndarray],
        aux_contour_list: List[np.ndarray],
):
    # Generate our mask frames. They should share the same xy dimensions.
    primary_mask = np.zeros_like(fluorescent_frame, dtype=np.uint8)
    aux_mask = np.zeros_like(fluorescent_frame, dtype=np.uint8)
    # We then draw our contours onto their respective masks.
    primary_mask = cv2.drawContours(primary_mask, primary_contour_list, -1, 255, cv2.FILLED)
    aux_mask = cv2.drawContours(aux_mask, aux_contour_list, -1, 255, cv2.FILLED)
    mask_combined = cv2.bitwise_and(primary_mask, aux_mask)
    mask_indices = np.where(mask_combined == 255)
    fluorescent_intensities = fluorescent_frame[mask_indices[0], mask_indices[1]]
    # debug_visualization(mask_combined, 'Non-Downsampled Aux Frame')
    # if np.mean(fluorescent_intensities) < 31:
    #     pass
    # else:
    #     new_cell = CellSignal(
    #         contour_index=contour_idx,
    #         sum_signal=np.sum(fluorescent_intensities),
    #         fluorescent_pixels=fluorescent_intensities,
    #     )
    #     cell_signals.append(new_cell)


def generate_arbitrary_stats(
        channel_stats: List[List[CellSignal]],
        bin_lower_threshold: int = 25,
):
    '''

    Args:
        channel_stats:

    Returns:

    '''
    out_list = []
    for channel_set in channel_stats:
        channel_list = []
        for cell_set in channel_set:
            raw_signal = [cell_st.sum_signal for cell_st in cell_set]
            hist, bins = np.histogram(raw_signal, bins=100)
            bin_limit = bins[bin_lower_threshold]
            filtered_set = list(filter(lambda x: x.sum_signal > bin_limit, cell_set))
            f_signal = [fil_st.sum_signal for fil_st in filtered_set]
            median_signal = np.median(f_signal)
            std_dev = np.std(f_signal)
            inner_cell_stats = CellStats(
                fl_signal=f_signal,
                raw_signal=raw_signal,
                median_signal=median_signal,
                std_dev=std_dev,
                filtered_len=len(filtered_set),
            )
            channel_list.append(inner_cell_stats)
        out_list.append(channel_list)
    return out_list


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
        hist, bins = np.histogram(c_signal, bins=100)
        bin_limit = bins[3]
        filtered_set = list(filter(lambda x: x.sum_signal < bin_limit, cell_set))
        f_signal = [fil_st.sum_signal for fil_st in filtered_set]
        median_signal = np.median(f_signal)
        std_dev = np.std(f_signal)
        out_list.append((median_signal, std_dev, len(filtered_set)))
    return out_list


def write_raw_frames(
        input_frames: List[np.ndarray],
        channel_names: List[str],
        output_fp: str,
        f_index: int
):
    '''

    Args:
        input_frames:
        channel_names:
        f_index:
        output_fp:

    Returns:

    '''
    output_fp += f'/raw/{f_index}'
    if not os.path.isdir(output_fp):
        os.makedirs(output_fp)
    for index, frame in enumerate(input_frames):
        flattened_arrays = list(map(np.ravel, frame))
        for channel, array in zip(channel_names, flattened_arrays):
            fp = output_fp + f'/{index}_{channel}_raw_data.csv'
            np.savetxt(fp, array, delimiter=",", fmt='%d')


def write_stat_record(
        input_package: TiffPackage,
        record_fp: str,
):
    num_channels = input_package.img_array.shape[0]
    num_observations = input_package.img_array.shape[1]
    fl_channels = num_channels - 1
    dt = {
        'index': list(range(num_observations))
    }
    for channel in range(fl_channels):
        # First result is PhLc
        channel_offset = channel + 1
        channel_name = input_package.channel_names[channel_offset]
        channel_fluorescence = []
        channel_std_dev = []
        channel_cell_count = []
        for i in range(num_observations):
            target_frame = input_package.img_array[channel_offset][i]
            # Calculate total fluorescence of the frame
            channel_fluorescence.append(np.sum(target_frame))
            channel_std_dev.append(np.std(target_frame))
            channel_cell_count.append(
                len(input_package.cell_signal_auxiliary_frames[channel][i])
            )
        dt[f'{channel_name}_fluorescence'] = channel_fluorescence
        dt[f'{channel_name}_std_dev'] = channel_std_dev
        dt[f'{channel_name}_cell_count'] = channel_cell_count
        df = pd.DataFrame.from_dict(dt)
        df.to_csv(record_fp)



