'''
--------------------------------------------------------------------------------
Description:
    Converts Nikon proprietary format microscopy images into constituent TIFFs
    for the purpose of image analysis and tracking. The images are exported in
    motion, channel, time_series.

How to Use:
    This assumes that you followed the installation instructions in the README
    to have a valid python environment locally in the repository.

    `$ poetry run python nd2_to_tiff.py --input_filepath <MY_DIR_WITH_ND2>`


Written by W.R. Jackson <wrjackso@bu.edu>, DAMP Lab 2020
--------------------------------------------------------------------------------
'''
import datetime
from typing import (
    List,
    Union,
)

import click
import cv2
import imageio
import numpy as np
import tqdm
from PIL import (
    ImageFont,
)
from pims import ND2_Reader as nd2_sdk


# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=RuntimeWarning)
# warnings.filterwarnings("ignore")


# ---------------------------- Utility Functions -------------------------------
def colorize_frame(input_frame: np.ndarray, color: str) -> np.ndarray:
    '''
    Converts a 1-Channel Grayscale Image to RGB Space and then converts the
    prior image to a singular color.

    Args:
        input_frame: Input numpy array
        color: (red|blue|green) What color to convert to:

    Returns:
        Colorized ndarray

    '''
    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_GRAY2RGB)
    if color == 'green':
        input_frame[:, :, (0, 2)] = 0
    if color == 'red':
        input_frame[:, :, (1, 2)] = 0
    if color == 'blue':
        input_frame[:, :, (0, 1)] = 0
    return input_frame


def downsample_image(input_frame: np.ndarray) -> np.ndarray:
    '''
    Downsamples a 16bit input image to an 8bit image. Will result in loss
    of signal fidelity due to loss of precision.

    Args:
        input_frame: Input numpy array

    Returns:
        Downsampled ndarray
    '''
    return (input_frame / 256).astype('uint8')


def modify_contrast(
        input_frame: np.ndarray,
        alpha: int = 5,
        gamma: int = 127,
):
    '''
    Modifies the contrast of the input image. This is done through modifying
    the alpha and gamma. Alpha multiplies the underlying intensity while the
    gamma is a straight increase to the intensity.

    Args:
        input_frame: Input numpy array
        alpha: Alpha Value
        gamma: Gamma Value

    Returns:
        Contrast Modified ndarray

    '''
    return cv2.addWeighted(
        input_frame,
        alpha,
        input_frame,
        0,
        gamma,
    )


def empty_frame_check(input_frame: np.ndarray) -> bool:
    '''
    Checks to see if the inputted frame is empty.

    An implementation detail from the underlying nd2reader library we're using:
    Discontinuous or dropped frames are represented by NaN values instead of
    zeroes which will later be implicitly cast to zeroes when being persisted
    to disk. There does seem to be some places where there is a mismatch
    between channels which results in the resultant image being off in terms
    of stride.

    Args:
        input_frame: Input numpy array

    Returns:
        Whether or not the frame is 'emtpy'

    '''
    nan_converted_frame = np.nan_to_num(input_frame)
    if np.sum(nan_converted_frame) == 0:
        return True
    return False


def apply_brightness_contrast(
        input_frame: np.ndarray,
        alpha=12.0,
        beta=0,
):
    '''
    Modifies the contrast of the input image. This is done through modifying
    the alpha and gamma. Alpha multiplies the underlying intensity while the
    gamma is a straight increase to the intensity. Basically, alpha is contrast
    and beta is brightness

    Args:
        input_frame: Input numpy array
        alpha: Alpha Value
        beta: Beta Value

    Returns:
        Contrast Modified ndarray

    '''
    return cv2.convertScaleAbs(
        input_frame,
        alpha=alpha,
        beta=beta,
    )


# -------------------------------- Statistics ----------------------------------


def generate_summary_statistics(
        c2_frame_and_name: List[Union[np.ndarray, str]],
        c3_frame_and_name: List[Union[np.ndarray, str]],
        frame_index: int,
) -> List[str]:
    '''
    Generates summary statistics for the output frame.

    Args:
        c2_frame_and_name: Channel 2 Frame and it's channel label
        c3_frame_and_name: Channel 3 Frame and it's channel label
        frame_index: Index of the frame. We perform modulus division to get
            the remainder outside the function as they should all be happening
            at the same time scale.

    Returns:
        List of summary statistics
    '''
    out_list = []
    c2_frame, c2_name = c2_frame_and_name
    c3_frame, c3_name = c3_frame_and_name
    minute_increment = frame_index // 10
    d1 = f'{datetime.timedelta(minutes=minute_increment * 15)}'
    out_list.append(f'{d1}')
    out_list.append(f'Frame Index {frame_index}')
    out_list.append(f'{c2_name} Mean Intensity: {int(np.mean(c2_frame))}')
    out_list.append(f'{c3_name} Mean Intensity: {int(np.mean(c3_frame))}')
    out_list_max_len = 0
    for line in out_list:
        if len(line) > out_list_max_len:
            out_list_max_len = len(line)
    for index, line in enumerate(out_list):
        padded_line = line.ljust(out_list_max_len, ' ')
        out_list[index] = padded_line
    return out_list


def write_summary_statistics(
        input_frame: np.ndarray,
        input_stats: List[str],
        position: str = 'br',
        font_name: str = 'Hack.ttf',
        font_size: int = 12,
):
    '''
    Writes the summary statistics for the frame.

    This is done via the generation of a new frame that has the text written
    on top of it being collapsed into our 'final' image to avoid clipping
    any of the data.

    Args:
        input_frame: Frame to be written on.
        input_stats: Input statistics
        position: Where to write the text at. (Kind of broken atm)
        font_name: Font resource. Used for determining offsets.
        font_size: Font size.

    Returns:
        Final image with statistics.
    '''
    initial_position = [0, 0]
    array_shape_x, array_shape_y = input_frame.shape[0], input_frame.shape[1]
    write_frame = np.zeros((array_shape_x, array_shape_y, 3), dtype=np.uint8)
    max_length = 0
    max_string = None
    for line in input_stats:
        if len(line) > max_length:
            max_length = len(line)
            max_string = line
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
        initial_position[0] = int((array_shape_x - offset_buffers[0]) - \
                                  max_length) * 2
        initial_position[1] = int(array_shape_y - string_heights + \
                                  offset_buffers[1]) + 150
    if position == 'bl':
        initial_position[0] = initial_position[0] + offset_buffers[0]
        initial_position[1] = array_shape_y + string_heights + offset_buffers[1]
    y_track = initial_position[1]
    for line in input_stats:
        line_width = font.getsize(line)[0]
        _x_pos = int((initial_position[0] - line_width) / 2)
        cv2.putText(
            write_frame,
            line,
            (_x_pos, y_track),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            [255, 255, 255],
            2,
            cv2.LINE_AA,
        )
        y_track = y_track - 30
    output_frame = cv2.addWeighted(
        input_frame,
        1,
        write_frame,
        1.5,
        0,
    )
    return output_frame


@click.command()
@click.option(
    '--input_file',
    prompt='Input path to directory containing ND2 Files',
    help='The path to the ND2 files to be converted.',
    default='/media/jackson/WD_BLACK/microscopy_images/agarose_pads/SR15_2mM_IPTG_Agarose_TS_3h_3.nd2',
)
def make_gif(input_file: str):
    with nd2_sdk(input_file) as input_frames:
        frame_count = input_frames.sizes['t']
        channels = input_frames.sizes['c']
        print(input_frames.sizes)
        input_frames.iter_axes = 'mtc'
        observation_start_stop = []
        number = frame_count * channels
        for observation in range(number):
            observation_start_stop.append(
                [
                    (observation * number),
                    (observation + 1) * number,
                    channels * channels,
                    f'{input_file.split("/")[-1][:-3]}_{observation}',
                    ]
            )
        channels = ['blah', 'bloo', 'thing']
        for start, stop, stride, label in observation_start_stop:
            interior_list = []
            try:
                for frame_index in tqdm.tqdm(
                        range(start, stop, stride),
                        desc=f'Converting {label} into gif...',
                ):
                    initial_frame = input_frames[frame_index]
                    c2_frame = input_frames[frame_index + 1]
                    c3_frame = input_frames[frame_index + 2]
                    if any(
                            map(
                                empty_frame_check,
                                [initial_frame, c2_frame, c3_frame]
                            )
                    ):
                        continue
                    stats_list = generate_summary_statistics(
                        [c2_frame, channels[1]],
                        [c3_frame, channels[2]],
                        frame_index % number,
                        )
                    initial_frame, c2_frame, c3_frame = map(
                        downsample_image,
                        [initial_frame, c2_frame, c3_frame],
                    )
                    # !!!
                    # IF YOU WANT TO CHANGE THE CONTRAST AND BRIGHTNESS DO IT
                    # IN THE BELOW FUNCTIONS.
                    c2_frame = apply_brightness_contrast(
                        c2_frame,
                        alpha=12,
                        beta=0,
                    )
                    c3_frame = apply_brightness_contrast(
                        c3_frame,
                        alpha=12,
                        beta=0,
                    )
                    initial_frame = colorize_frame(
                        initial_frame,
                        'grey',
                    )
                    c2_frame = colorize_frame(
                        c2_frame,
                        'red',
                    )
                    c3_frame = colorize_frame(
                        c3_frame,
                        'green',
                    )
                    intermediate_frame = cv2.addWeighted(
                        initial_frame,
                        1,
                        c2_frame,
                        0.75,
                        0,
                    )
                    out_frame = cv2.addWeighted(
                        intermediate_frame,
                        1,
                        c3_frame,
                        0.75,
                        0,
                    )
                    out_frame = write_summary_statistics(
                        out_frame,
                        stats_list,
                    )
                    for i in range(4):
                        interior_list.append(out_frame)
                imageio.mimsave(f'{label}.gif', interior_list)
            except IndexError:
                exit(0)


if __name__ == '__main__':
    make_gif()
