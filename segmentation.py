'''
--------------------------------------------------------------------------------
Description:

Roadmap:

Written by W.R. Jackson <wrjackso@bu.edu>, DAMP Lab 2020
--------------------------------------------------------------------------------
'''
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

import imageio
import numpy as np
import tqdm
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from server.signal_processing import (
    signal_transform,
    sigpro_utility,
    graphing,
    stats,
)

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
    # if current_device_mask_frame is None:
    #     current_device_mask_frame, sentinel_val = \
    #         signal_transform.device_highpass_filter(grey_frame)
    # if not signal_transform.mask_recalculation_check(grey_frame, sentinel_val):
    #     current_device_mask_frame, sentinel_val = \
    #         signal_transform.device_highpass_filter(grey_frame)
    # grey_frame = signal_transform.normalize_subsection(
    #     grey_frame,
    #     current_device_mask_frame,
    #     sentinel_val,
    # )
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
    return out_frame, frame_stats, mask_frame, raw_frame

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
    # if os.path.exists(out_structure):
    #     shutil.copyfile(
    #         out_structure,
    #         f'output/{title}_f{index}_prior.gif',
    #     )
    imageio.imwrite(out_structure, frame)
    return out_structure


def test_timing_based_seperation(master_frame_list, title):
    for experiment_num, file_locs in enumerate(master_frame_list):
        input_frames = sigpro_utility.ingress_tiffs(file_locs)
        final_frame = []
        frame_count = len(input_frames)
        empty_stats = [[(0, 0, 0), (0, 0, 0)]] * frame_count
        final_stats = deque(empty_stats, maxlen=frame_count)
        import time
        start = time.time()
        # segmentation_pipeline(input_frames)
        with Pool(PROCESSES) as p:
            results = list(
                tqdm.tqdm(
                    p.imap(segmentation_pipeline, input_frames),
                    total=len(input_frames),
                    desc='Running Segmentation Pipeline...')
            )
            p.close()
            p.join()
        # Final Frame, stats, mask, and Raw
        print('Finished Segmentation Pipeline! Parsing Results..')
        # stop = time.time() - start
        # print(f'Total Time: {stop}'
        #       f'Approx. Time per frame = {stop / len(input_frames)}'
        #       )
        for index, result in enumerate(tqdm.tqdm(results)):
            out_frame, frame_stats, mask_frame, raw_frame = result
            # canvas_list.append(out_frame)
            final_frame.append(out_frame)
            final_stats.append(frame_stats)
        appended_results = []
        for index, result in enumerate(tqdm.tqdm(results)):
            result = list(result)
            point_in_time = list(islice(final_stats, 0, index))
            historical_point = copy.copy(point_in_time)
            delta = len(final_stats) - index
            historical_point += [[(0, 0, 0), (0, 0, 0)]] * delta
            result.append(historical_point)
            result.append(title)
            result.append(index)
            appended_results.append(result)
        with Pool(PROCESSES) as p:
            c_results = list(tqdm.tqdm(p.imap(iter_create_canvas, appended_results), total=len(input_frames)))
            p.close()
            p.join()
        # for canvas_list.append(graphing.generate_image_canvas(
        #         out_frame,
        #         signal_transform.downsample_image(raw_frame),
        #         final_stats,
        #         f'output/{title}- Frame: {index}',
        #         frame_count
        #     ))
        # sigpro_utility.display_frame(c_results[0])
        print('Result Parsing Finished! Saving to disk, this might take a second...')
        stats.write_stat_record(
            final_stats,
            f'output/{title}_{datetime.datetime.now().date()}.csv'
        )
        write_list = []
        for index, res in enumerate(c_results):
            write_list.append([res, title, index])
        # iter = list(sigpro_utility.list_chunking(canvas_list, 20))
        # file_list = []
        # TODO: I need to balance moving things out of memory in the main python
        #  function with the speed loss of hitting the disk for every file.
        with Pool(PROCESSES) as p:
            file_handles = list(
                tqdm.tqdm(
                    p.imap(
                        write_out,
                        write_list,
                    ), total=len(input_frames), desc="Writing out File..."))
            p.close()
            p.join()
        # while True:
        #     res = next(iter, None)
        #     if res is None:
        #         break
        #     out_structure = f'output/{title}_f{file_counter}.gif'
        #     if os.path.exists(out_structure):
        #         shutil.copyfile(
        #             out_structure,
        #             f'output/{title}_f{file_counter}_prior.gif',
        #         )
        #     imageio.mimsave(out_structure, res)
        #     file_list.append(out_structure)
        #     file_counter += 1
        # Compression
        # for file in file_handles:
        #     sigpro_utility.resize_gif(file)
        # # Stick em together
        final_output = []
        for file in file_handles:
            final_output.append(imageio.imread(file))
            os.remove(file)
        imageio.mimsave(f'output/{title}_{experiment_num}.gif', final_output)


def enqueue_segmentation_pipeline(input_frames: List[np.ndarray], title: str):
    input_frames = sigpro_utility.ingress_tiffs(input_frames)
    final_frame = []
    frame_count = len(input_frames)
    empty_stats = [[(0, 0, 0), (0, 0, 0)]] * frame_count
    final_stats = deque(empty_stats, maxlen=frame_count)
    results = list(tqdm.tqdm(map(segmentation_pipeline, input_frames), total=len(input_frames)))
    # Final Frame, stats, mask, and Raw
    print('Finished Segmentation Pipeline! Parsing Results..')
    for index, result in enumerate(tqdm.tqdm(results)):
        out_frame, frame_stats, mask_frame, raw_frame = result
        # canvas_list.append(out_frame)
        final_frame.append(out_frame)
        final_stats.append(frame_stats)
    appended_results = []
    for index, result in enumerate(tqdm.tqdm(results)):
        result = list(result)
        point_in_time = list(islice(final_stats, 0, index))
        historical_point = copy.copy(point_in_time)
        delta = len(final_stats) - index
        historical_point += [[(0, 0, 0), (0, 0, 0)]] * delta
        result.append(historical_point)
        result.append(title)
        result.append(index)
        appended_results.append(result)
    c_results = list(tqdm.tqdm(map(iter_create_canvas, appended_results), total=len(input_frames)))
    print('Result Parsing Finished! Saving to disk, this might take a second...')
    stats.write_stat_record(
        final_stats,
        f'output/{title}_{datetime.datetime.now().date()}.csv'
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
            ), total=len(input_frames)))
    final_output = []
    for file in file_handles:
        final_output.append(imageio.imread(file))
        os.remove(file)
    imageio.mimsave(f'output/{title}.gif', final_output)


@click.command()
@click.option(
    '--fn',
    default="data/raw_nd2/1-SR_1_5_6hPre-C_2h_1mMIPTG_OFF_1hmMIPTG_ON_22hM9_TS_MC1.nd2",
    prompt='Filepath to Input ND2 files.'
)
@click.option(
    '--subselection',
    default=0,
    prompt='Subselection of Frames. 0 indicates that segmentation will be '
           'performed on all frames'
)
def segmentation_ingress(fn: str, subselection: int):
    title = (fn.split('/')[-1])[:-4]
    print(f'Loading {title}... (This may take a second)')
    frames = sigpro_utility.test_iter_axes_options(fn)
    # frames = sigpro_utility.ingress_tiffs(frames)
    print(f'Loading Complete!')
    if subselection:
        frames = frames[:subselection]
    print(f'Starting Segmentation Pipeline..')
    for f_index, frame in enumerate(frames):
        enqueue_segmentation_pipeline(frame, f'{title}_{f_index}')



if __name__ == '__main__':
    segmentation_ingress()