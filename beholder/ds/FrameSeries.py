'''
--------------------------------------------------------------------------------
Description:

Roadmap:

Written by W.R. Jackson <wrjackso@bu.edu>, DAMP Lab 2020
--------------------------------------------------------------------------------
'''
import os
from typing import (
    List,
)

import cv2
import nd2reader
import numpy as np

from beholder.utils.logging import BLogger

VALID_FILE_EXTS = ['.nd2']

COLORATION_LUT = {
    'm-Cherry': 'red',
    'PhC': 'grey',
    'YFP': 'green'
}

LOG = BLogger()


def transparent_frame(
        input_frame: np.ndarray,
        color_to_convert=(0, 0, 0),
):
    # Convert to RGB with alpha channel
    output = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGBA)

    # Coinvert from BGR to BGRA
    bgra = cv2.cvtColor(input_frame, cv2.COLOR_BGR2BGRA)

    # Slice of alpha channel
    alpha = bgra[:, :, 3]

    # Use logical indexing to set alpha channel to 0 where BGR=0
    alpha[np.all(bgra[:, :, 0:3] == color_to_convert, 2)] = 0

    return output

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


async def retrieve_frame(
        fov_num: int,
        frame_num: int,
        channel_num: int,
) -> np.ndarray:
    # Only fetching the grey frame for now.
    fs = FrameSeries()
    if fs.is_active():
        return fs.get_renderable_frame(
            fov_num,
            frame_num,
            channel_num,
        )


async def retrieve_colored_frame(
        fov_num: int,
        frame_num: int,
        channel_num: int,
) -> np.ndarray:
    # Only fetching the grey frame for now.
    fs = FrameSeries()
    if fs.is_active():
        return fs.get_colored_frame(
            fov_num,
            frame_num,
            channel_num,
        )


async def fetch_fov_size():
    fs = FrameSeries()
    if fs.is_active():
        return len(fs.frame_sets)
    return 0


async def fetch_xy_size(fov_num):
    fs = FrameSeries()
    LOG.info('Function Starts')
    if fs.is_active():
        print(fov_num)
        active_frames = fs.frame_sets[fov_num]
        return len(active_frames.frames)
    return 0


async def load_frame_series(fp: str):
    '''

    Args:
        fp:

    Returns:

    '''
    filename, file_extension = os.path.splitext(fp)
    LOG.info(f'Loading File {filename}...')
    fs = FrameSeries()
    fs.load(fp)
    LOG.info(f'Successfully Loaded {filename}...')


class SingletonBaseClass(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class FrameSingular:
    # There's some stuff that we need to think about vis a vis the idea of
    # having a massive dataset loaded into operational memory but we'll figure
    # that out later.

    def __init__(
            self,
            frame_data: np.ndarray,
            channel_labels=List[str],
    ):
        # We pass it a 3 dimensional numpy array consisting of the three
        # channels. This should work with the three dimensional constructs, but
        # we should be cognizant of that.
        self.frame_data = frame_data
        self.channel_labels = channel_labels

    def get_frame(self):
        return self.frame_data[0]


class FrameSet:

    def __init__(
            self,
            frame_iterable: List[np.ndarray],
            channel_labels: List[str],
    ):
        # This could probably just be a list or any other form of collection,
        # but I feel like you are going to want a mapping later on.
        self.frames = {}
        for f_index, frame in enumerate(frame_iterable):
            self.frames[f_index] = FrameSingular(frame, channel_labels)

    def get_frame(self, frame_num: int, channel_num: int) -> FrameSingular:
        return self.frames[frame_num].frame_data[channel_num]

    def get_frame_api(self, fov_num: int):
        frame = self.frames[fov_num]
        raw = frame.frame_data[0]
        return raw.tostring()

    def get_frame_shape(self):
        frame = self.frames[0].frame_data[0]
        return frame.shape[0], frame.shape[1]

    def get_frame_dtype(self):
        frame = self.frames[0].frame_data[0]
        return frame.dtype


class FrameSeries(metaclass=SingletonBaseClass):

    def __init__(self):
        self.filepath = None
        self.frame_sets = {}
        self.channels = None

    def load(self, filepath: str):
        filename, file_extension = os.path.splitext(filepath)
        if file_extension not in VALID_FILE_EXTS:
            raise RuntimeError(
                f'{file_extension} not a valid file extension. Currently '
                f'accepted filetypes are {VALID_FILE_EXTS}'
            )
        self.filepath = filepath
        with nd2reader.ND2Reader(self.filepath) as input_frames:
            input_frames.iter_axes = 'vtc'
            view_number = input_frames.sizes['v']
            time_scale = input_frames.sizes['t']
            num_channels = input_frames.sizes['c']
            channels = input_frames.metadata['channels']
            self.channels = channels
            frame_sets = []
            access_toggle = False
            access_value = 0
            for i in range(view_number):
                frame_offset = i * time_scale
                inner_list = []
                for j in range(0, time_scale * num_channels, num_channels):
                    if access_toggle:
                        print('Access Toggle Triggered')
                        access_index = frame_offset + j - access_value
                        access_toggle = False
                        access_value = 0
                    else:
                        access_index = frame_offset + j
                    grey_frame = input_frames[access_index]
                    ch1_frame = input_frames[access_index + 1]
                    ch2_frame = input_frames[access_index + 2]
                    if any(
                            map(
                                empty_frame_check,
                                [grey_frame, ch1_frame, ch2_frame]
                            )
                    ):
                        continue
                    else:
                        d_stack = np.stack([grey_frame, ch1_frame, ch2_frame])
                        inner_list.append(d_stack)
                if len(inner_list) <= 10:
                    access_toggle = True
                    access_value = len(inner_list)
                if len(inner_list) > 10:
                    print(len(inner_list))
                    frame_sets.append(inner_list)
                else:
                    continue
            for set_index, frame_set in enumerate(frame_sets):
                self.frame_sets[set_index] = FrameSet(
                    frame_set,
                    channels,
                )

    def is_active(self):
        return True if self.frame_sets else False

    def get_individual_frame(self, fov_num: int, frame_num: int):
        '''

        Args:
            fov_num:
            frame_num:

        Returns:

        '''
        fov = self.frame_sets[fov_num]
        return fov[frame_num].get_frame(frame_num)

    def get_renderable_frame(self, fov_num, frame_num, channel_num):
        if self.frame_sets is None:
            LOG.info('Please load dataset.')
        fov = self.frame_sets[fov_num]
        return fov.get_frame(frame_num, channel_num)

    def get_colored_frame(self, fov_num, frame_num, channel_num):
        frame = self.get_renderable_frame(fov_num, frame_num, channel_num)
        channel_name = self.channels[channel_num]
        # TODO. We might have to do hardcore alpha for the dark
        return self.colorize_frame(
            input_frame=frame,
            color=COLORATION_LUT[channel_name],
        )

    def get_frame_shape(self):
        if self.frame_sets is None:
            LOG.info('Please load dataset.')
        fov = self.frame_sets[0]
        return fov.get_frame_shape()

    def get_frame_dtype(self):
        if self.frame_sets is None:
            LOG.info('Please load dataset.')
        fov = self.frame_sets[0]
        return fov.get_frame_dtype()

    def colorize_frame(self, input_frame: np.ndarray, color: str) -> np.ndarray:
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
        if color == 'blue':
            input_frame[:, :, (0, 1)] = 0
        if color == 'red':
            input_frame[:, :, (1, 2)] = 0
        return input_frame


if __name__ == '__main__':
    load_frame_series('/mnt/shared/code/damp_lab/beholder/data/raw_nd2/3-SR_1_5_4h_Pre-C_1h_1mMIPTG_After2h_MCM1.nd2')
