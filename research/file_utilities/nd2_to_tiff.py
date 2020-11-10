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

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

@click.command()
@click.option(
    '--input_filepath',
    prompt='Input path to directory containing ND2 Files',
    help='The path to the ND2 files to be converted.',
    default='../example_files',
)
@click.option(
    '--output_filepath',
    prompt='Output directory for converted TIFF Files',
    help='The path to the ND2 files to be converted.',
    default='../test',
)
@click.option(
    '--visualize',
    prompt='Visualize Images during conversion?',
    help='Whether or not to display the images while they are happening..',
    default=False,
)
def converter(input_filepath: str, output_filepath: str, visualize: bool):
    for input_file in glob.glob(f'{input_filepath}/*.nd2'):
        # Just for future reference:
        # The new reader (nd2reader) can't process the images but it can parse
        # the metadata. The older reader (ND2_Reader) can process the metadata
        # but not the images. They both have separate `malloc` errors and are
        # developed by the same guy. ¯\_(ツ)_/¯
        if not os.path.isdir(input_filepath):
            print(
                'Unable to locate directed input filepath. Please check '
                'filepath.'
            )
            exit(1)
        if not os.path.isdir(output_filepath):
            print(
                'Unable to locate output filepath. Please check output '
                'filepath.'
            )
            exit(1)
        with ND2_Reader(input_file) as input_frames:
            with nd2reader.ND2Reader(input_file) as meta_data:
                # Available dimensions are:
                #   - x: Width
                #   - y: Height
                #   - c: Channel 3
                #   - t: Time 30
                #   - m: Movie? 9
                # Should be 810 frames.
                frame_count = meta_data.sizes['t']
                fov_count = meta_data.sizes['v']
                channels = meta_data.sizes['c']
                input_frames.iter_axes = 'mtc'
                file_name_list = []
                for frame in range(frame_count):
                    for fov in range(fov_count):
                        for channel in range(channels):
                            file_name_list.append([frame, fov, channel])
                for index, input_frame in enumerate(tqdm.tqdm(input_frames)):
                    output_filename = file_name_list[index]
                    frame, fov, channel = output_filename
                    frame_number = f'{frame}'.zfill(6)
                    file_name = f't{frame_number}xy{fov}c{channel}'
                    if visualize:
                        cv2.imshow('img', input_frame)
                        time.sleep(.1)
                    # This means we have to output TIFFs
                    if channel > 0:
                        try:
                            val = filters.threshold_otsu(input_frame)
                            thresholded_array = input_frame < val
                            thresholded_array = np.logical_not(thresholded_array)
                        except ValueError:
                            thresholded_array = input_frame
                        png_save(
                            f'{output_filepath}/masks/{file_name}.png',
                            thresholded_array,
                        )
                    tiff_save(
                        f'{output_filepath}/tiffs/{file_name}.tiff',
                        input_frame,
                    )
                if visualize:
                    cv2.destroyAllWindows()




if __name__ == '__main__':
    converter()
