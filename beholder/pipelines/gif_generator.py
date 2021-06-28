'''
--------------------------------------------------------------------------------
Description:

Roadmap:

Written by W.R. Jackson <wrjackso@bu.edu>, DAMP Lab 2020
--------------------------------------------------------------------------------
'''
import glob
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import (
    List,
)

import cv2
import imageio
import numpy as np
import tqdm

from beholder.signal_processing import (
    apply_brightness_contrast,
    downsample_image,
    colorize_frame,
    combine_frame,
    modify_contrast,
    jump_color,
)
from beholder.signal_processing.sigpro_utility import (
    get_channel_and_wl_data_from_xml_metadata,
    ingress_tiff_file,
)
from beholder.utils import (
    BLogger,
    get_analysis_location,
    convert_channel_name_to_color,
)

LOG = BLogger()


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def enqueue_panel_based_gif_generation(
        input_datasets: List[str],
        runlist_fp: str,
        alpha: int,
        beta: int,
):
    """

    Args:
        input_datasets:
        runlist_fp:
        alpha:
        beta:

    Returns:

    """
    panel_fp_map = {}
    channel_list = []
    output_path = get_analysis_location(runlist_fp)
    for panel_index, dataset_fp in tqdm.tqdm(
            enumerate(input_datasets),
            desc=f'Enumerating over datasets and parsing tiffs...',
            total=len(input_datasets),
    ):
        tiff_dir_root = os.path.join(dataset_fp, 'raw_tiffs')
        metadata_root = os.path.join(dataset_fp, 'metadata.xml')
        tree = ET.parse(metadata_root)
        channels = get_channel_and_wl_data_from_xml_metadata(tree)
        channel_list.append(channels)
        resultant_tiffs = glob.glob(f'{tiff_dir_root}/*.tiff')
        for tiff_fp in resultant_tiffs:
            panel_label = Path(tiff_fp).stem
            if panel_label not in panel_fp_map:
                panel_fp_map[panel_label] = []
            panel_fp_map[panel_label].append(tiff_fp)
    final_dest = os.path.join(
        output_path,
        'panel_gifs'
    )
    channels = [channel[1] for channel in channel_list[0][0]]
    if not os.path.isdir(final_dest):
        os.makedirs(final_dest)
    for panel_index, panel_entry in tqdm.tqdm(
            enumerate(panel_fp_map),
            desc=f'Building Panel Gifs...',
            total=len(panel_fp_map)
    ):
        tiff_list = []
        panel_lst = panel_fp_map[panel_entry]
        for panel_fp in panel_lst:
            tiff_list.append(ingress_tiff_file(panel_fp))
        # We now have a list of all the tiff files, now we need to create a list
        # or ndarray of each of the little slices after we do channel based
        # compositing.
        master_lst = []
        for tiff in tiff_list:
            frame_tracker = 1
            primary_channel = tiff[0]
            auxiliary_channels = tiff[1:]
            # I don't think I need to downsample because all I need are the
            # resultant images.
            # I think this is kosher due to the loss in dimensionality.
            inner_list = []
            for frame_index in range(primary_channel.shape[0]):
                base_frame = primary_channel[frame_index]
                base_frame = colorize_frame(base_frame, color='grey')

                for channel_index in range(auxiliary_channels.shape[0]):
                    channel_name = channels[channel_index + 1]
                    color_name = convert_channel_name_to_color(
                        channel_name
                    )
                    color_frame = auxiliary_channels[channel_index][frame_index]
                    # color_frame = apply_brightness_contrast(
                    #     input_frame=auxiliary_channels[channel_index][frame_index],
                    #     alpha=2,
                    #     beta=0,
                    # )
                    # base_frame = np.uint8(base_frame)
                    color_frame = colorize_frame(color_frame, color_name)
                    # color_frame = modify_contrast(color_frame)
                    color_frame = jump_color(color_frame, color_name, 3)
                    # color_frame = increase_brightness(color_frame)
                    base_frame = cv2.addWeighted(
                        base_frame,
                        1,
                        color_frame,
                        0.75,
                        0,
                    )
                    base_frame = combine_frame(base_frame, color_frame)
                frame_tracker += 1
                write_frame = np.zeros(
                    (
                        base_frame.shape[0],
                        base_frame.shape[1],
                        3
                    ),
                    dtype=np.uint16)
                x_pos = int((base_frame.shape[0] - 2000))
                y_pos = int((base_frame.shape[1] - 2000))
                cv2.putText(
                    write_frame,
                    f'Frame: {frame_tracker}',
                    (int(write_frame.shape[0]/2), int(write_frame.shape[1]/2)),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    [0, 0, 0],
                    2,
                    cv2.LINE_AA,
                )
                base_frame = cv2.addWeighted(
                    base_frame,
                    1,
                    write_frame,
                    1.5,
                    0,
                )
                master_lst.append(base_frame)
        gif_fp = os.path.join(
            final_dest,
            f'{panel_index}.gif'
        )
        imageio.mimsave(gif_fp, master_lst)
