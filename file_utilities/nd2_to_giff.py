'''
--------------------------------------------------------------------------------
Description:
    Converts Nikon proprietary format microscopy images into constituent TIFFs
    for the purpose of image analysis and tracking. The images are exported in
    motion, channel, time_series.

How to Use:
    This assumes that you followed the installation instructions in the README
    to have a valid python environment locally in the repository.

    `$ poetry run python nd2_to_tiff.py --input_filepath <MY_DIR_WITH_ND2>
    --output_filepath <MY_OUTPUT_DIR> --visualize <IF_YOU_WANT_VISUALIZATION>`


Written by W.R. Jackson <wrjackso@bu.edu>, DAMP Lab 2020
--------------------------------------------------------------------------------
'''
import glob
import os
import time
import warnings

import click
import cv2
import nd2reader
import numpy as np
import tqdm
from pims import ND2_Reader
from skimage import filters
from skimage.io import imsave as png_save
from tiffile import imsave as tiff_save
import imageio

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def colorize_frame(input_image, color):
    input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
    if color == 'green':
        input_image[:, :, (0, 2)] = 0
    if color == 'red':
        input_image[:, :, (1, 2)] = 0
    return input_image


@click.command()
@click.option(
    '--input_file',
    prompt='Input path to directory containing ND2 Files',
    help='The path to the ND2 files to be converted.',
    default='/media/jackson/WD_BLACK/microscopy_images/to_gif/10_h.nd2',
)
@click.option(
    '--output_file',
    prompt='Output directory for converted TIFF Files',
    help='The path to the ND2 files to be converted.',
    default='../test',
)
def make_gif(input_file: str, output_file: str):
    with nd2reader.ND2Reader(input_file) as input_frames:
        # Available dimensions are:
        #   - x: Width
        #   - y: Height
        #   - c: Channel 3
        #   - t: Time 30
        #   - m: Movie? 9
        # Should be 810 frames.
        frame_count = input_frames.sizes['t']
        channels = input_frames.sizes['c']
        fov_count = input_frames.sizes['v']
        input_frames.iter_axes = 'vtc'
        file_name_list = []
        # Take in the image
        # Iterate over each view, and if that view is blank skip over
        # it.
        # If the view is real, we overlay both green and red channels
        # over the grayscale image.
        # We then append the resultant image to a list
        # Once we've iterated over the sum of the entire image, we
        # collapse it into a gif.
        # Start: 0
        # Stop: 288
        # Start two: 360 ish
        # Stop 2: 729 ish
        # Start: 1089
        # Stop: 1452
        # Start: 1815
        # Stop: 2172
        # Start: 2537
        # Stop: 2901
        # Start: 3261
        # Start:  3627
        observation_start_stop = [
            [0, 288, 9, 'test_1'],
            [288, 360, 9, 'test_2'],
            [360, 729, 9, 'test_3'],
            [729, 1089, 9, 'test_4'],
            [1089, 1452, 9, 'test_5'],
            [1452, 1815, 9, 'test_6'],
            [2172, 2537, 9, 'test_7'],
            [2901, 3261, 9, 'test_8'],
            [3261, 3627, 9, 'test_9'],
        ]
        for start, stop, stride, label in observation_start_stop:
            interior_list = []
            for frame_index in range(start, stop, stride):
                grey_frame = colorize_frame(
                    input_frames[frame_index],
                    False,
                )
                cherry_frame = colorize_frame(
                    input_frames[frame_index + 1],
                    'green',
                )
                yfp_frame = colorize_frame(
                    input_frames[frame_index + 2],
                    'red',
                )
                intermediate_frame = cv2.addWeighted(
                    grey_frame,
                    1,
                    cherry_frame,
                    0.76,
                    0,
                )
                out_frame = cv2.addWeighted(
                    intermediate_frame,
                    1,
                    yfp_frame,
                    0.75,
                    0,
                )
                position = (
                    int(out_frame.shape[1] / 2 - 268 / 2),
                    int(out_frame.shape[0] / 2 - 36 / 2)
                )
                cv2.putText(
                    out_frame,  # numpy array on which text is written
                    f'{frame_index % stride}',  # text
                    position,  # position at which writing has to start
                    cv2.FONT_HERSHEY_DUPLEX,  # font family
                    1,  # font size
                    (209, 80, 0, 255),  # font color
                    3)  # font stroke
                interior_list.append(out_frame)
            imageio.mimsave(f'{label}.gif', interior_list)


if __name__ == '__main__':
    make_gif()