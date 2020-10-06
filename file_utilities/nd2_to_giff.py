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
import datetime
import warnings
from typing import (
    List,
)

import click
import cv2
import imageio
import nd2reader
import numpy as np
import tqdm
from PIL import (
    ImageFont,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")


# ---------------------------- Utility Functions -------------------------------
def dtype_conversion_frame(
        input_image: np.ndarray,
        datatype: str = 'uint8',
):
    if datatype == 'uint8':
        return input_image.astype(np.uint8)


def colorize_frame(input_image, color):
    input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
    if color == 'green':
        input_image[:, :, (0, 2)] = 0
    if color == 'red':
        input_image[:, :, (1, 2)] = 0

    return input_image


# Modify the `contrast` value below to change the contrast of the input
# pictures.
def modify_contrast(
        input_image, alpha: int = 5,
        contrast: int = 127,
):
    '''

    Args:
        input_image:
        alpha:
        contrast:

    Returns:

    '''
    return cv2.addWeighted(
        input_image,
        alpha,
        input_image,
        0,
        contrast,
    )


# -------------------------------- Statistics ----------------------------------


def generate_summary_statistics(
        timing_information: datetime.datetime,
        cherry_frame: np.ndarray,
        yfp_frame: np.ndarray,
        frame_index: int,
):
    out_list = []
    minute_increment = frame_index // 10
    # Time from Zero
    d1 = timing_information + datetime.timedelta(minutes=minute_increment*15)
    # Figure out how to make this more human readable. HH:MM?
    out_list.append(f'{d1}')
    # Segmentation Count goes here?
    out_list.append(f'Frame Index {frame_index}')
    out_list.append(f'Green Mean Intensity: {int(np.mean(cherry_frame))}')
    out_list.append(f'Red Mean Intensity: {int(np.mean(yfp_frame))}')
    return out_list


def histogram_visualization():
    '''

    Returns:

    '''
    pass


def write_summary_statistics(
        input_frame: np.ndarray,
        input_stats: List[str],
        position: str = 'mid',
        font_name: str = 'Hack.ttf',
        font_size: int = 12,
):
    initial_position = [0, 0]
    array_shape_x, array_shape_y = input_frame.shape[0], input_frame.shape[1]
    downward_text = True if position in ('tl', 'tr') else False
    # Determine our maximum line length. Assuming that requested text is
    # centered.
    max_length = 0
    max_string = None
    for line in input_stats:
        if len(line) > max_length:
            max_length = len(line)
            max_string = line
    # Convert our characters into pixels for the purposes of offsetting the
    # origin.
    font = ImageFont.truetype(font_name, font_size)
    max_string_width, line_height = font.getsize(max_string)
    string_heights = max_string_width * len(input_stats)
    offset_buffers = array_shape_x / 10, array_shape_y / 10
    if position == 'tl':
        initial_position[0] = initial_position[0] + offset_buffers[0]
        initial_position[1] = initial_position[1] + offset_buffers[1]
    if position == 'mid':
        initial_position[0] = array_shape_x / 2 - max_string_width
        initial_position[1] = array_shape_y / 2 + string_heights
    if position == 'tr':
        initial_position[0] = (array_shape_x - offset_buffers[0]) - max_length
        initial_position[1] = array_shape_y + offset_buffers[1]
    if position == 'br':
        initial_position[0] = (array_shape_x - offset_buffers[0]) - max_length
        initial_position[1] = array_shape_y + string_heights + offset_buffers[1]
    if position == 'bl':
        initial_position[0] = initial_position[0] + offset_buffers[0]
        initial_position[1] = array_shape_y + string_heights + offset_buffers[1]
    y_track = 100
    out_frame = input_frame
    for line in input_stats:
        line_width = font.getsize(line)[0]
        _x_pos = (initial_position[0] - line_width) / 2
        cv2.putText(
            out_frame,
            line,
            (100, y_track),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            0,
            6,
            cv2.LINE_AA,
        )
        y_track = y_track + 30
    return out_frame


@click.command()
@click.option(
    '--input_file',
    prompt='Input path to directory containing ND2 Files',
    help='The path to the ND2 files to be converted.',
    default='/media/jackson/WD_BLACK/microscopy_images/to_gif/10_h.nd2',
)

def make_gif(input_file: str):
    with nd2reader.ND2Reader(input_file) as input_frames:
        frame_count = input_frames.sizes['t']
        channels = input_frames.sizes['c']
        input_frames.iter_axes = 'vtc'
        observation_start_stop = []
        number = frame_count * channels
        print(input_frames.get_timesteps())
        for observation in range(number-1):
            observation_start_stop.append(
                [
                    (observation * number),
                    (observation + 1) * number,
                    channels * channels,
                    f'{input_file.split("/")[-1][:-3]}_{observation}',
                ]
            )
        events = input_frames.events
        for start, stop, stride, label in tqdm.tqdm(observation_start_stop):
            interior_list = []
            for frame_index in range(start, stop, stride):
                grey_frame = input_frames[frame_index]
                cherry_frame = input_frames[frame_index + 1]
                yfp_frame = input_frames[frame_index + 2]
                date = input_frames.metadata['date']
                stats_list = generate_summary_statistics(
                    date,
                    cherry_frame,
                    yfp_frame,
                    frame_index,
                )
                grey_frame = colorize_frame(
                    grey_frame,
                    False,
                )
                grey_frame = modify_contrast(
                    grey_frame,
                    alpha=5,
                    contrast=127,
                )
                cherry_frame = colorize_frame(
                    cherry_frame,
                    'green',
                )
                cherry_frame = modify_contrast(
                    cherry_frame,
                    alpha=5,
                    contrast=127,
                )
                yfp_frame = colorize_frame(
                    yfp_frame,
                    'red',
                )
                yfp_frame = modify_contrast(
                    yfp_frame,
                    alpha=5,
                    contrast=127,
                )
                intermediate_frame = cv2.addWeighted(
                    grey_frame,
                    1,
                    cherry_frame,
                    0.75,
                    0,
                )
                out_frame = cv2.addWeighted(
                    intermediate_frame,
                    1,
                    yfp_frame,
                    0.75,
                    0,
                )
                out_frame = write_summary_statistics(
                    out_frame,
                    stats_list,
                )
                interior_list.append(out_frame)
            print(f'{label}.gif')
            imageio.mimsave(f'{label}.gif', interior_list)


if __name__ == '__main__':
    make_gif()
