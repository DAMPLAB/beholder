'''
--------------------------------------------------------------------------------
Description:

Roadmap:

Written by W.R. Jackson <wrjackso@bu.edu>, DAMP Lab 2020
--------------------------------------------------------------------------------
'''
import datetime
import struct

import cv2
import matplotlib.pyplot as plt
import numpy as np
import nd2reader
from pims import ND2_Reader as nd2_sdk


# --------------------------- Utility Functionality ----------------------------
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

def get_channel_names(fp: str):
    with nd2reader.ND2Reader(fp) as input_frames:
        return input_frames.metadata['channels']

def parse_nd2_file(fn: str):
    out_list = []
    try:
        with nd2reader.ND2Reader(fn) as input_frames:
            input_frames.iter_axes = 'vtc'
            for start_idx in range(0, len(input_frames), 3):
                grey_frame = input_frames[start_idx]
                ch1_frame = input_frames[start_idx + 1]
                ch2_frame = input_frames[start_idx + 2]
                if any(
                        map(
                            empty_frame_check,
                            [grey_frame, ch1_frame, ch2_frame]
                        )
                ):
                    continue
                else:
                    out_list.append((grey_frame, ch1_frame, ch2_frame))
    # Try/except logic to spackle over some of the weird indexing issues
    # that seem to occur for three dimensional images vice two dimensional
    except KeyError:
        with nd2_sdk(fn) as input_frames:
            input_frames.iter_axes = 'mtc'
            for start_idx in range(0, len(input_frames), 3):
                grey_frame = input_frames[start_idx]
                ch1_frame = input_frames[start_idx + 1]
                ch2_frame = input_frames[start_idx + 2]
                if any(
                        map(
                            empty_frame_check,
                            [grey_frame, ch1_frame, ch2_frame]
                        )
                ):
                    continue
                else:
                    out_list.append((grey_frame, ch1_frame, ch2_frame))
    return out_list


def open_microscopy_image(filename: str) -> np.ndarray:
    '''

    Args:
        filename:

    Returns:

    '''
    file_ext = filename.split('.')[-1]
    # We should probably be smart about picking up the coloring/channels because
    # we implicitly assume things exist within greyscale 1-channel due to our
    # nd2 context.
    if file_ext == 'tiff' or file_ext == 'tif':
        out_image = cv2.imread(filename)
        return out_image
    # I think they don't persist nd2s to disk until writes are complete.
    if file_ext == 'nd2':
        try:
            with nd2reader.ND2Reader(filename) as input_frames:
                if len(input_frames) > 1:
                    # Make words good.
                    print('Input ND2 Files more than one image.')
                return input_frames[0], input_frames[1], input_frames[2]
        # Try/except logic to spackle over some of the weird indexing issues
        # that seem to occur for three dimensional images vice two dimensional
        except KeyError:
            with nd2_sdk(filename) as input_frames:
                input_frames.iter_axes = 'mtc'
                return input_frames[0], input_frames[1], input_frames[2]


def get_initial_image_nd2(input_nd2_file: str) -> np.ndarray:
    '''
        Gets the initial grayscale frame from an ND2 file.

        Args:
            input_nd2_file:

        Returns:

        '''
    with nd2reader.ND2Reader(input_nd2_file) as input_frames:
        return input_frames[0]


def display_frame(
        input_frame: np.ndarray,
        image_label: str = 'beholder',
        context: str = 'cv2',
):
    '''

        Args:
            input_frame:
            image_label:
            context:

        Returns:

        '''
    if context == 'jupyter':
        plt.imshow(
            input_frame,
            cmap='gray',
            interpolation='nearest',
        )
        return
    if context == 'cv2':
        architecture_bit = struct.calcsize('P') * 8
        if architecture_bit == 64:
            exit_func = cv2.waitKey(0) & 0xFF
        else:
            exit_func = cv2.waitKey(0)
        cv2.imshow(image_label, input_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # k = exit_func
        # Pressing the escape key exits the pop-up window, pressing s saves the
        # frame
        # if k == 27:
        #     cv2.destroyAllWindows()
        # elif k == ord('s'):  # wait for 's' key to save and exit
        #     cv2.imwrite(f'{image_label}_{datetime.date}.png', input_frame)
        #     cv2.destroyAllWindows()
