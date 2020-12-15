'''
--------------------------------------------------------------------------------
Description:

Roadmap:

Written by W.R. Jackson <wrjackso@bu.edu>, DAMP Lab 2020
--------------------------------------------------------------------------------
'''
import csv
from collections import deque
import click
import copy
import datetime
import multiprocessing as mp
import os
import shutil
from typing import (
    List,
    Optional,
    Tuple,
)
import warnings
from functools import partial
from itertools import repeat, islice
from multiprocessing import Pool, freeze_support
from pathlib import Path
import gc
import imageio
import numpy as np
import pandas as pd
import tqdm
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import matplotlib.pyplot as plt

from backend.signal_processing import (
    signal_transform,
    sigpro_utility,
    graphing,
    stats,
)
from memory_profiler import profile

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")

PROCESSES = mp.cpu_count() - 2


def calculate_attic(
        fp: str,
        swatch_point_0: Tuple[int, int],
        swatch_point_1: Tuple[int, int],
        mask_point_0: Tuple[int, int],
        mask_point_1: Tuple[int, int],
):
    '''

    Args:
        fp:
        swatch_point_0:
        swatch_point_1:
        mask_point_0:
        mask_point_1:

    Returns:

    '''
    reference_frame = sigpro_utility.get_initial_image_nd2(fp)
    swatch_median = np.median(signal_transform.crop_from_points(
        reference_frame,
        swatch_point_0,
        swatch_point_1,
    )
    )
    mask_median = np.median(signal_transform.crop_from_points(
        reference_frame,
        mask_point_0,
        mask_point_1,
    )
    )
    return abs(swatch_median + mask_median)


def preprocess_initial_grey_and_find_contours(initial_frame: np.ndarray):
    # Each image transform should be giving us back an np.ndarray of the same
    # shape and size.
    out_frame = signal_transform.percentile_threshold(initial_frame)
    out_frame = signal_transform.invert_image(out_frame)
    out_frame = signal_transform.remove_background(out_frame)
    out_frame = signal_transform.downsample_image(out_frame)
    contours = signal_transform.find_contours(out_frame)
    return contours


def preprocess_initial_color_and_find_contours(initial_frame: np.ndarray):
    # Each image transform should be giving us back an np.ndarray of the same
    # shape and size.
    out_frame = signal_transform.downsample_image(initial_frame)
    out_frame = signal_transform.apply_brightness_contrast(
        out_frame,
        alpha=2,
        beta=0,
    )
    out_frame = signal_transform.percentile_threshold(out_frame, 80, 98)
    out_frame = signal_transform.invert_image(out_frame)
    out_frame = signal_transform.remove_background(out_frame)
    out_frame = signal_transform.downsample_image(out_frame)
    contours = signal_transform.find_contours(out_frame)
    return contours


def preprocess_color_channel(
        initial_frame: np.ndarray,
        color: str,
        alpha: float = 12,
        beta: int = 0,
):
    out_frame = signal_transform.downsample_image(initial_frame)
    out_frame = signal_transform.apply_brightness_contrast(
        out_frame,
        alpha,
        beta,
    )
    out_frame = signal_transform.colorize_frame(out_frame, color)
    return out_frame


def contour_filtration(contours):
    # TODO: We then need to refine our approach in terms of segmentation either
    #  via eroding or some other mechanism. I think edge delineation is being
    #  confounded by the lack of depth in the microscopy image and the
    #  microfluidic device it's being housed in.
    # filtered_contours = signal_transform.cellular_highpass_filter(contours)
    # return filtered_contours
    return contours


def generate_mask(input_frame: np.ndarray, contours):
    out_frame = graphing.draw_mask(
        input_frame,
        contours,
        colouration='rainbow',
    )
    return out_frame


def segmentation_pipeline(
        input_frames: Tuple[np.ndarray, np.ndarray, np.ndarray],
):
    grey_frame, red_frame, green_frame = input_frames
    raw_frame = copy.copy(grey_frame)
    c_red_frame = preprocess_color_channel(red_frame, 'red')
    c_green_frame = preprocess_color_channel(green_frame, 'green')
    mask_frame = np.zeros_like(grey_frame)
    contours = preprocess_initial_grey_and_find_contours(grey_frame)
    contours = contour_filtration(contours)
    green_cell_signals = stats.fluorescence_detection(
        grey_frame,
        green_frame,
        contours,
    )
    red_cell_signals = stats.fluorescence_detection(
        grey_frame,
        red_frame,
        contours,
    )
    frame_stats = stats.generate_frame_stats(
        green_cell_signals,
        red_cell_signals,
    )
    labeled_green = graphing.label_cells(
        signal_transform.downsample_image(green_frame),
        contours,
        green_cell_signals,
    )
    labeled_red = graphing.label_cells(
        signal_transform.downsample_image(red_frame),
        contours,
        red_cell_signals,
    )
    labeled_green = signal_transform.colorize_frame(labeled_green, 'green')
    labeled_red = signal_transform.colorize_frame(labeled_red, 'red')
    d_grey_frame = signal_transform.downsample_image(grey_frame)
    out_frame = signal_transform.combine_frame(
        d_grey_frame,
        c_red_frame,
    )
    out_frame = signal_transform.combine_frame(
        out_frame,
        labeled_red,
    )
    out_frame = signal_transform.combine_frame(
        out_frame,
        c_green_frame,
    )
    out_frame = signal_transform.combine_frame(
        out_frame,
        labeled_green,
    )
    mask_frame = generate_mask(mask_frame, contours)
    return out_frame, frame_stats, mask_frame, raw_frame, green_cell_signals, red_cell_signals

    # TODO: This is untenable for a live mode, so we're going to have to make
    # some duplicated code at some point


def iter_create_canvas(result):
    out_frame, frame_stats, mask_frame, raw_frame, final_stats, title, index = result
    return graphing.generate_image_canvas(
        out_frame,
        signal_transform.downsample_image(raw_frame),
        final_stats,
        f'{title}- Frame: {index}',
    )


def write_out(entry):
    frame, title, index = entry
    out_structure = f'output/{title}_f{index}.gif'
    imageio.imwrite(out_structure, frame)
    return out_structure


def enqueue_segmentation_pipeline(
        input_frames: List[np.ndarray],
        title: str,
        channel_names: List[str],
        f_index: int,
        render_videos: bool,
):
    if os.path.getsize(input_frames) < 400:
        return
    input_frames = sigpro_utility.ingress_tiffs(input_frames)
    # channel_lengths = int(len(input_frames)/len(channel_names))
    channel_lengths = 3
    frames = []
    grey = input_frames[0]
    red = input_frames[1]
    green = input_frames[2]
    input_frames = list(zip(grey, red, green))
    # for i in range(0, channel_lengths):
    #     frames.append(
    #         tuple([
    #             input_frames[i],
    #             input_frames[i+channel_lengths],
    #             input_frames[i+(channel_lengths*2)]])
    #     )
    # input_frames = frames
    final_frame = []
    frame_count = len(input_frames)
    empty_stats = [[(0, 0, 0), (0, 0, 0)]] * frame_count
    final_stats = deque(empty_stats, maxlen=frame_count)
    root = f'output/{title}_{datetime.datetime.now().date()}'
    if not os.path.exists(root):
        os.mkdir(root)
    output_directory = root + f'/{f_index}'
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    with Pool(PROCESSES) as p:
        results = list(
            tqdm.tqdm(p.imap(segmentation_pipeline, input_frames),
                      total=len(input_frames)))
    # Final Frame, stats, mask, and Raw
    segmentation_output_chan_1 = f'{output_directory}/{f_index}_{channel_names[1]}.csv'
    segmentation_output_chan_2 = f'{output_directory}/{f_index}_{channel_names[2]}.csv'
    if not os.path.exists(segmentation_output_chan_1):
        Path(segmentation_output_chan_1).touch()
    if not os.path.exists(segmentation_output_chan_2):
        Path(segmentation_output_chan_2).touch()
    for index, result in enumerate(tqdm.tqdm(results)):
        _, _, _, _, green_stats, red_stats = result
        with open(segmentation_output_chan_1, 'a+') as chan_1_file:
            output = [stat.sum_signal for stat in green_stats]
            writer = csv.writer(chan_1_file)
            writer.writerow(output)
        with open(segmentation_output_chan_2, 'a+') as chan_2_file:
            output = [stat.sum_signal for stat in red_stats]
            writer = csv.writer(chan_2_file)
            writer.writerow(output)
    for index, result in enumerate(tqdm.tqdm(results, desc="Extracting Results...")):
        out_frame, frame_stats, mask_frame, raw_frame, _, _ = result
        # canvas_list.append(out_frame)
        final_frame.append(out_frame)
        final_stats.append(frame_stats)
    appended_results = []
    for index, result in enumerate(tqdm.tqdm(results, desc="Parsing Results...")):
        result = result[:-2]
        result = list(result)
        point_in_time = list(islice(final_stats, 0, index))
        historical_point = copy.copy(point_in_time)
        delta = len(final_stats) - index
        historical_point += [[(0, 0, 0), (0, 0, 0)]] * delta
        result.append(historical_point)
        result.append(title)
        result.append(index)
        appended_results.append(result)
    if render_videos:
        with Pool(PROCESSES) as p:
            c_results = list(tqdm.tqdm(p.imap(iter_create_canvas, appended_results),
                                       desc="Generating Visualizations...",
                                       total=len(input_frames)))
        stats.write_stat_record(
            final_stats,
            f'{output_directory}/{f_index}_results.csv'
        )
        write_list = []
        for index, res in enumerate(c_results):
            write_list.append([res, title, index])
        # iter = list(sigpro_utility.list_chunking(canvas_list, 20))
        # file_list = []
        # TODO: I need to balance moving things out of memory in the main python
        #  function with the speed loss of hitting the disk for every file.
        file_handles = list(
            tqdm.tqdm(
                map(
                    write_out,
                    write_list,
                ),
                desc="Writing Everything to disk...",
                total=len(input_frames)))
        final_output = []
        for file in file_handles:
            final_output.append(imageio.imread(file))
            os.remove(file)
        imageio.mimsave(f'{output_directory}/{f_index}_video.gif', final_output)


# /mnt/shared/data/microscopy/4-SR_1_9_16hIPTG_6hM9_TS_MC5.nd2
@click.command()
@click.option(
    '--fp',
    default='/mnt/shared/code/damp_lab/beholder/data/raw_tiffs/',
    prompt='Filepath to Input ND2 file or TIFF Directory.'
)
@click.option(
    '--subselection',
    default=0,
    prompt='Subselection of Frames. 0 indicates that segmentation will be '
           'performed on all frames'
)
@click.option(
    '--render_videos',
    default=True,
    prompt='Render Output Videos'
)
def segmentation_ingress(fp: str, subselection: int, render_videos: bool):
    title = (fp.split('/')[-1])[:-4]
    print(f'Loading {title}... (This may take a second)')
    frame_paths = None
    # channel_names = None
    # if Path(fp).suffix == '.nd2':
    #     print('ND2 Detected. Converting into TIFF format...')
    #     frame_paths = sigpro_utility.nd2_convert(fp)
    #     channel_names = sigpro_utility.get_channel_names(fp)
    if os.path.isdir(fp):
        # frame_paths = sigpro_utility.batch_convert(fp)
        print('Tiff Mode Activated...')
        frame_paths = sigpro_utility.grab_tiff_filenames(fp)
        channel_names = ['PhlC', 'm-Cherry', 'YFP']
    if frame_paths is not None:
        print(f'Loading Complete!')
        if subselection:
            frame_paths = frame_paths[:subselection]
        print(f'Starting Segmentation Pipeline..')
        for f_index in range(0, len(frame_paths)):
            frame = frame_paths[f_index]
            enqueue_segmentation_pipeline(
                frame,
                f'{Path(frame).name.split(".")[0]}',
                channel_names,
                f_index,
                render_videos,
            )
            print('-----')
            plt.close('all')
            gc.collect()


if __name__ == '__main__':
    segmentation_ingress()
