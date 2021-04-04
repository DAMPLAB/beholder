'''
--------------------------------------------------------------------------------
Description:

Roadmap:

Written by W.R. Jackson <wrjackso@bu.edu>, DAMP Lab 2020
--------------------------------------------------------------------------------
'''
import glob
import math
import operator
import os
from pathlib import Path
import struct
from xml.etree import ElementTree as ETree

import bioformats as bf
import cv2
import javabridge
import matplotlib.pyplot as plt
import numpy as np
import nd2reader
from pims import ND2_Reader as nd2_sdk
from PIL import Image
import tqdm
import tiffile
from beholder.utils.slack_messaging import slack_message

# javabridge.start_vm(class_path=bf.JARS)


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


def glob_tiffs(fp: str, chunk_size: int, channel_num: int = 3):
    '''

    Args:
        fp:
        chunk_size

    Returns:

    '''
    tiffs = glob.glob(f'{fp}/*.tiff')
    tiffs = sorted(tiffs, key=lambda x: int(((x.split('/')[-1]).split('_')[-1])[:-5]))
    master_list = []
    for i in range(0, len(tiffs), channel_num * chunk_size):
        chunk_list = []
        for j in range(i, i + chunk_size):
            chunk_list.append(tiffs[j])
        master_list.append(chunk_list)
    return master_list


def ingress_tiffs(input_fn: str):
    tiff = tiffile.imread(input_fn)
    uint16_cast = (tiff * 65536).round().astype(np.uint16)
    return uint16_cast

def get_separated_frames(fn: str):
    channels = get_channel_names(fn)
    base_filename = (fn.split("/")[-1])[:-4]
    # Should have either the ability to quanitfy the amount of memory that this
    # will require or pass some sort of flag from a higher context which will
    # define disk based file behavior.
    dir_path = f'data/raw_tiffs/{base_filename}'
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    with nd2reader.ND2Reader(fn) as input_frames:
        if len(input_frames.sizes) < 5:
            slack_message(f'Could not find 5th dimension for {fn}')
            return None
        print(f'{input_frames.sizes=}')
        input_frames.iter_axes = 'vtc'
        view_number = input_frames.sizes['v']
        time_scale = input_frames.sizes['t']
        num_channels = input_frames.sizes['c']
        outer_list = []
        for i in range(view_number):
            frame_offset = i * time_scale
            inner_list = []

            for j in range(0, (time_scale*num_channels), num_channels):
                access_index = frame_offset + j
                print(f'{access_index=}')
                print(f'{time_scale*num_channels=}')
                print(f'{j=}')
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
                    write_list = []
                    for k, w_frame in zip(range(3), [grey_frame, ch1_frame, ch2_frame]):
                        fn = f'data/raw_tiffs/{base_filename}/{channels[k]}_{access_index}.tiff'
                        if not os.path.exists(fn):
                            tiffile.imsave(fn, w_frame)
                        write_list.append(fn)
                    inner_list.append(write_list)
            if len(inner_list) > 10:
                outer_list.append(inner_list)
            else:
                continue
        return outer_list

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


def generate_iter_frames(fn: str):
    '''

    Args:
        fn:

    Returns:

    '''
    out_list = []
    try:
        with nd2reader.ND2Reader(fn) as input_frames:
            input_frames.bundle_axes = 'xyv'
            input_frames.iter_axes = 'tc'
            for xy_pos in range(input_frames.sizes['v']):
                pass
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


def generate_frame_timing_groupings(fn: str):
    out_list = []
    try:
        with nd2reader.ND2Reader(fn) as input_frames:
            input_frames.bundle_axes = 'xyv'
            input_frames.iter_axes = 'tc'
            loop_data = input_frames.metadata['experiment']['loops']
            frame_spread = []
            current_count = 0
            c_dim = input_frames.sizes['c']
            v_dim = input_frames.sizes['v']
            factor = c_dim * v_dim
            for loop in loop_data:
                durr = loop['duration']
                samp_i = loop['sampling_interval']
                frame_length = math.floor(durr / samp_i) * factor
                prior_frame = current_count
                current_count += frame_length
                frame_spread.append([prior_frame, current_count])
            for frame_slices in frame_spread:
                start, stop = frame_slices
                internal_frames = input_frames[start: stop]
                internal_list = []
                for start_idx in range(0, len(internal_frames), 3):
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
                        internal_list.append((grey_frame, ch1_frame, ch2_frame))
                out_list.append(internal_list)
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


def parse_xml_metadata(xml_string, array_order='tyxc'):
    array_order = array_order.upper()
    names, sizes, resolutions = [], [], []
    spatial_array_order = [c for c in array_order if c in 'XYZ']
    size_tags = ['Size' + c for c in array_order]
    res_tags = ['PhysicalSize' + c for c in spatial_array_order]
    metadata_root = ETree.fromstring(xml_string)
    for child in metadata_root:
        if child.tag.endswith('Image'):
            names.append(child.attrib['Name'])
            for grandchild in child:
                if grandchild.tag.endswith('Pixels'):
                    att = grandchild.attrib
                    sizes.append(tuple([int(att[t]) for t in size_tags]))
                    resolutions.append(tuple([float(att[t])
                                              for t in res_tags]))
    return names, sizes, resolutions



def tiff_splitter(fp: str):
    frames = parse_nd2_file(fp)
    channels = get_channel_names(fp)
    base_filename = (fp.split("/")[-1])[:-4]
    # Creating a Tiff File for each channel.
    for i, frame in enumerate(tqdm.tqdm(frames)):
        dir_path = f'data/raw_tiffs/{base_filename}'
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        for j, inner_frame in enumerate(frame):
            fn = f'data/raw_tiffs/{base_filename}/{channels[j]}_{i}.tiff'
            tiffile.imsave(fn, frame)

def batch_convert(target_directory: str):
    file_paths = glob.iglob(target_directory + '**/*.nd2', recursive=True)
    files_and_sizes = ((path, os.path.getsize(path)) for path in file_paths)
    sorted_files_with_size = sorted(files_and_sizes, key=operator.itemgetter(1))
    tiff_directories = []
    for fp, _ in sorted_files_with_size:
        tiff_directories.append(nd2_convert(fp))
    return [item for sublist in tiff_directories for item in sublist]

def grab_tiff_filenames(target_directory: str):
    thing = list(glob.iglob(target_directory + '**/*.tiff', recursive=True))
    return thing

def nd2_convert(fp: str, output_directory: str = 'data/raw_tiffs'):
    javabridge.start_vm(class_path=bf.JARS)
    base_name = Path(fp).name
    channel_dir = f'{output_directory}/{base_name}'
    if not os.path.isdir(channel_dir):
        os.mkdir(channel_dir)
    md = bf.get_omexml_metadata(fp)
    rdr = bf.ImageReader(fp, perform_init=True)
    names, sizes, resolutions = parse_xml_metadata(md)
    # We assume uniform shape + size for all of our input frames.
    num_of_frames, x_dim, y_dim, channels = sizes[0]
    print(2)
    for i in range(len(names)):
        output_array = []
        for j in range(num_of_frames):
            blank_check = rdr.read(c=1, t=j, series=i)
            if np.sum(blank_check) == 0:
                continue
            else:
                channel_array = []
                for k in range(channels):
                    temp = rdr.read(c=k, t=j, series=i)
                    channel_array.append(temp)
                output_array.append(channel_array)
        out_array = np.asarray(output_array)
        print(len(out_array.shape))
        if len(out_array.shape) == 4:
            out_array = out_array.transpose(1, 0, 2, 3)
        if len(out_array.shape) == 3:
            out_array = out_array.transpose(1, 0, 2)
        tiffile.imsave(f'{channel_dir}/{base_name}_{i}.tiff', out_array)
    # javabridge.kill_vm()
    return channel_dir