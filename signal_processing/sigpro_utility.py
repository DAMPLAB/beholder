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
from PIL import Image

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


def get_fov(fp: str):
    with nd2reader.ND2Reader(fp) as input_frames:
        return len(list(input_frames.metadata['fields_of_view']))

def list_chunking(input_list, chunk_size):
    for i in range(0, len(input_list), chunk_size):
        yield input_list[i:i + chunk_size]

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


def sublist_splitter(primary_list, max_count, divisor):
    for i in range(0, max_count, divisor):
        yield primary_list[i:i + divisor]

def resize_gif(path, save_as=None, resize_to=None):
    """
    Resizes the GIF to a given length:

    Args:
        path: the path to the GIF file
        save_as (optional): Path of the resized gif. If not set, the original gif will be overwritten.
        resize_to (optional): new size of the gif. Format: (int, int). If not set, the original GIF will be resized to
                              half of its size.
    """
    all_frames = extract_and_resize_frames(path, resize_to)

    if not save_as:
        save_as = path

    if len(all_frames) == 1:
        print("Warning: only 1 frame found")
        all_frames[0].save(save_as, optimize=True)
    else:
        all_frames[0].save(save_as, optimize=True, save_all=True, append_images=all_frames[1:], loop=1000)


def analyseImage(path):
    """
    Pre-process pass over the image to determine the mode (full or additive).
    Necessary as assessing single frames isn't reliable. Need to know the mode
    before processing all frames.
    """
    im = Image.open(path)
    results = {
        'size': im.size,
        'mode': 'full',
    }
    try:
        while True:
            if im.tile:
                tile = im.tile[0]
                update_region = tile[1]
                update_region_dimensions = update_region[2:]
                if update_region_dimensions != im.size:
                    results['mode'] = 'partial'
                    break
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    return results


def extract_and_resize_frames(path, resize_to=None):
    """
    Iterate the GIF, extracting each frame and resizing them

    Returns:
        An array of all frames
    """
    mode = analyseImage(path)['mode']

    im = Image.open(path)

    if not resize_to:
        resize_to = (im.size[0] // 2, im.size[1] // 2)

    i = 0
    p = im.getpalette()
    last_frame = im.convert('RGBA')

    all_frames = []

    try:
        while True:
            # print("saving %s (%s) frame %d, %s %s" % (path, mode, i, im.size, im.tile))

            '''
            If the GIF uses local colour tables, each frame will have its own palette.
            If not, we need to apply the global palette to the new frame.
            '''
            if not im.getpalette():
                im.putpalette(p)

            new_frame = Image.new('RGBA', im.size)

            '''
            Is this file a "partial"-mode GIF where frames update a region of a different size to the entire image?
            If so, we need to construct the new frame by pasting it on top of the preceding frames.
            '''
            if mode == 'partial':
                new_frame.paste(last_frame)

            new_frame.paste(im, (0, 0), im.convert('RGBA'))

            new_frame.thumbnail(resize_to, Image.ANTIALIAS)
            all_frames.append(new_frame)

            i += 1
            last_frame = new_frame
            im.seek(im.tell() + 1)
    except EOFError:
        pass

    return all_frames

if __name__ == '__main__':
    resize_gif('WithoutFilterRemoval.gif')