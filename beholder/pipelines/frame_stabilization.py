'''
--------------------------------------------------------------------------------
Description:

Roadmap:

Written by W.R. Jackson <wrjackso@bu.edu>, DAMP Lab 2020
--------------------------------------------------------------------------------
'''
import glob
import multiprocessing as mp
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import (
    List,
)

import cv2
import numpy as np
import tqdm
import imageio
import matplotlib.pyplot as plt

from beholder.ds import (
    TiffPackage,
)
from beholder.signal_processing.signal_transform import (
    unsharp_mask,
    eight_bit_plane_slice,
)
from beholder.signal_processing.sigpro_utility import (
    get_channel_and_wl_data_from_xml_metadata,
    ingress_tiff_file,
)
from beholder.signal_processing.stats import (
    debug_visualization,
)
from beholder.utils.config import (
    get_max_processes,
    do_single_threaded,
    do_visualization_debug,
    do_test_write,
)


def moving_average(curve, radius):
    window_size = 2 * radius + 1
    f = np.ones(window_size) / window_size
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    curve_smoothed = curve_smoothed[radius:-radius]
    return curve_smoothed


def curve_smoothing(trajectory: np.ndarray) -> np.ndarray:
    smoothed_trajectory = np.copy(trajectory)
    for i in range(3):
        smoothed_trajectory[:, i] = moving_average(trajectory[:, i], radius=10)
    # This is a copy+paste from their docs and I don't really get it. I'd
    # assume that this math here would have no effect on the signal.
    diff = smoothed_trajectory - trajectory
    smooth_transform = trajectory + diff
    return smooth_transform

def frame_prepro(channel_index: int, frame_index: int):
    pass


def visualize_stabilization(
        packaged_tiff: TiffPackage,
        transformation_list: List,
):
    previous_frame: np.ndarray = packaged_tiff.get_raw_frame(0, 0)
    # if previous_frame.dtype != np.uint8:
    #     previous_frame = downsample_image(previous_frame)
    # I think this is presupposing a 3-dimensional shift in space but due to
    # the housing mechanism of the microfluidic it should only be 2D.
    debug_visualization(previous_frame, 'Initial Frame')
    # They should implicitly correlate. Might want an assertion here.

    for index, transform in enumerate(transformation_list):
        image_t = cv2.warpAffine(
            previous_frame,
            transform, (
                packaged_tiff.get_frame_height(),
                packaged_tiff.get_frame_width(),
            )
        )
        debug_visualization(image_t, f'Image {index}')
        previous_frame: np.ndarray = packaged_tiff.get_raw_frame(
            0,
            index + 1,
        )


def write_stabilization(
        packaged_tiff: TiffPackage,
        transformation_list: List,
):
    previous_frame: np.ndarray = packaged_tiff.get_raw_frame(0, 0)
    write_list = [previous_frame]
    for index, transform in enumerate(transformation_list):
        image_t = cv2.warpAffine(
            previous_frame,
            transform, (
                packaged_tiff.get_frame_height(),
                packaged_tiff.get_frame_width(),
            )
        )
        print(np.sum(image_t))
        write_list.append(image_t)
        previous_frame: np.ndarray = packaged_tiff.get_raw_frame(
            0,
            index + 1,
        )
    print(len(write_list))
    return write_list


def stabilization_pipeline(
        packaged_tiff: TiffPackage,
):
    """
    Source:
        https://learnopencv.com/video-stabilization-using-point-feature-matching-in-opencv/
    Args:
        packaged_tiff:

    Returns:

    """
    num_frames = packaged_tiff.get_num_frames()
    frame_h = packaged_tiff.get_frame_height()
    frame_w = packaged_tiff.get_frame_width()
    # Get the (Frame Index 0) first (Channel 0) grey frame
    previous_frame: np.ndarray = np.uint8(packaged_tiff.get_raw_frame(0, 0))
    # if previous_frame.dtype != np.uint8:
    #     previous_frame = downsample_image(previous_frame)
    # I think this is presupposing a 3-dimensional shift in space but due to
    # the housing mechanism of the microfluidic it should only be 2D.
    final_bit_frame = eight_bit_plane_slice(previous_frame)[7]
    transform_list = []
    for i in range(num_frames - 1):
        # Detect feature points in previous frame
        prev_pts = cv2.goodFeaturesToTrack(
            previous_frame,
            maxCorners=25,
            qualityLevel=0.1,
            minDistance=10,
            blockSize=3,
        )
        corners = np.int0(prev_pts)
        for i in corners:
            x, y = i.ravel()
            cv2.circle(previous_frame, (x, y), 25, 255, -1)
            plt.imshow(previous_frame), plt.show()

        current_frame = unsharp_mask(np.uint8(packaged_tiff.get_raw_frame(0, i + 1)))
        # if previous_frame.dtype != np.uint8:
        #     previous_frame = downsample_image(previous_frame)
        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            previous_frame,
            current_frame,
            prev_pts,
            None,
        )

        # Sanity check
        assert prev_pts.shape == curr_pts.shape

        # Filter only valid points
        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        # Find transformation matrix
        m, _ = cv2.estimateAffinePartial2D(
            prev_pts,
            curr_pts,
        )
        transform_list.append(m)

        # Move to next frame
        previous_frame = current_frame
    # There's probably a filtering phase here.
    # The rotation and shearing is far too much for this example, so we're going
    # to reduce the values.
    # I'm not getting a lot of info about this, so this is an assumption about
    # the location of the shear coefficents.
    for transform in transform_list:
        transform[0][2] = 0
        transform[1][2] = 1
    transform_list = list(map(curve_smoothing, transform_list))
    if do_visualization_debug():
        visualize_stabilization(
            packaged_tiff=packaged_tiff,
            transformation_list=transform_list,
        )

    if do_test_write():
        write_list = write_stabilization(
            packaged_tiff=packaged_tiff,
            transformation_list=transform_list,
        )
        imageio.mimsave(f'{packaged_tiff.tiff_index}_stabilization_video.gif', write_list)
        for index, image in enumerate(write_list):
            imageio.imsave(f'{packaged_tiff.tiff_index}-{index}-stab-frame.png', image)
    # Then some write out occurs of some flavor.


def enqueue_frame_stabilization(input_fp: str):
    # We should have a top level metadata xml file and then we have a directory
    # called raw_tiffs that has all of the stuff that we really need to work on.
    # We need to take the xml file and extract the channels, sizes, and
    # resolutions and use that to create a class object that can encapuslate the
    # logic related to segmenting tiffs of various dimensions and properties.
    metadata_fp = os.path.join(input_fp, 'metadata.xml')
    tree = ET.parse(metadata_fp)
    # This assumes that everyone has the same amount of channels.
    # If we get to the point where ND2 files have different channels WITHIN
    # themselves I'm throwing my computer into the Charles...
    channels = get_channel_and_wl_data_from_xml_metadata(tree)
    packaged_tiffs = []
    tiff_path = os.path.join(input_fp, 'raw_tiffs')
    tiff_fp = glob.glob(tiff_path + '**/*.tiff')
    sorted_tiffs = sorted(tiff_fp, key=lambda x: int(Path(x).stem))
    # ------------------------------- PACKAGING TIFFS  -------------------------
    for index, tiff_file in tqdm.tqdm(
            enumerate(sorted_tiffs),
            total=len(sorted_tiffs),
            desc="Packaging Tiffs"
    ):
        array = ingress_tiff_file(tiff_file)
        if not array.shape[0]:
            continue
        wavelengths = [x[0] for x in channels[index]]
        channel_names = [x[1] for x in channels[index]]
        title = Path(tiff_file).stem
        inner_pack = TiffPackage(
            img_array=array,
            tiff_name=title,
            channel_names=channel_names,
            channel_wavelengths=wavelengths,
            tiff_index=index,
        )
        packaged_tiffs.append(inner_pack)
    # ---------------------------- PERFORM SEGMENTATION  -----------------------
    output_location = os.path.join(input_fp, 'segmentation_output')
    if not os.path.exists(output_location):
        os.mkdir(output_location)
    if do_single_threaded():
        stabilzation_results = list(
            tqdm.tqdm(
                map(
                    stabilization_pipeline,
                    packaged_tiffs
                ),
                total=len(packaged_tiffs),
                desc="Performing Frame Shift Calculation..."
            )
        )
    else:
        with mp.get_context("spawn").Pool(processes=get_max_processes()) as pool:
            segmentation_results = list(
                tqdm.tqdm(
                    pool.imap(stabilization_pipeline, packaged_tiffs),
                    total=len(packaged_tiffs)
                )
            )
