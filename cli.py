'''
--------------------------------------------------------------------------------
Description:

TODO:
    - [x] Batch Conversion Fixed. Have it be an alternative path in Typer.
    - [x] Persist metadata from conversion of ND2 to file in output directory
        per split ND2 file.
    - [ ] Plumb FrameSeries together with our new way of determining microscopy
            data from ImageJ. We should also use this to get the indices and
            what not from the channel names.
    - [ ] Put together the pipeline and actually make it run.

Written by W.R. Jackson <wrjackso@bu.edu>, DAMP Lab 2020
--------------------------------------------------------------------------------
'''
import copy
import csv
import dataclasses
import functools
import glob
import multiprocessing as mp
import operator
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import subprocess
import sys
from typing import (
    List,
    Tuple,
)

import bioformats as bf
import javabridge
import numpy as np
import tqdm
import typer
from rich.console import Console
from simple_term_menu import TerminalMenu

from beholder.signal_processing import (
    apply_brightness_contrast,
    cellular_highpass_filter,
    combine_frame,
    colorize_frame,
    downsample_image,
    clahe_filter,
    percentile_threshold,
    invert_image,
    erosion_filter,
    remove_background,
    normalize_frame,
    unsharp_mask,
    find_contours,
    generate_arbitrary_stats,
    label_cells,
    draw_mask,
    fluorescence_detection,
    generate_image_canvas,
    generate_segmentation_visualization,
)
from beholder.signal_processing.sigpro_utility import (
    nd2_convert,
    get_channel_data_from_xml_metadata,
    ingress_tiff_file,
)

console = Console()
app = typer.Typer()

RENDER_VIDEOS = True
PROCESSES = mp.cpu_count() - 1
COLOR_LUT = {
    'PhC': 'grey',
    'm-Cherry': 'red',
    'DAPI1': 'blue',
    'YFP': 'yellow',
}
DEBUG = True


# ------------------------------- Datastructures -------------------------------
@dataclasses.dataclass
class TiffPackage:
    img_array: np.ndarray
    tiff_name: str
    channel_names: List[str]
    channel_wavelengths: List[str]
    processed_array: List[np.ndarray] = None
    processed_frame_correlation: List[Tuple] = None
    output_statistics: List[Tuple] = None
    labeled_frames: List[np.ndarray] = None
    final_frames: List[np.ndarray] = None
    mask_frames: List[np.ndarray] = None

    processed_primary_frames: List = None
    processed_auxiliary_frames: List = None

    cell_signal_auxiliary_frames: List = None

    labeled_auxiliary_frames: List = None

    frame_stats: List = None

    stat_file_location: List = None

    def __post_init__(self):
        self.processed_array = []
        self.processed_primary_frames = []
        self.processed_auxiliary_frames = []
        self.cell_signal_auxiliary_frames = []
        self.labeled_auxiliary_frames = []
        self.final_frames = []
        self.mask_frames = []
        self.frame_stats = []
        self.stat_file_location = []

    def get_num_frames(self):
        return self.img_array.shape[1]


@dataclasses.dataclass
class StatisticResults:
    img_array: np.ndarray
    tiff_name: str
    channels: List[str]
    processed_array: List[np.ndarray] = None


# ----------------------- Command Line Utility Functions -----------------------
def beholder_text(input_text: str, color: str = '#49306B'):
    """

    Args:
        input_text:
        color:

    Returns:

    """
    console.print(input_text, style=color)


def validate_dir_path(input_fp: str):
    if not os.path.isdir(input_fp):
        raise RuntimeError(
            f'Unable to locate {input_fp}. Please check input and try again.'
        )


def generate_mask(input_frame: np.ndarray, contours):
    out_frame = draw_mask(
        input_frame,
        contours,
        colouration='rainbow',
    )
    return out_frame


# ---------------------------- Segmentation Pipeline ---------------------------
def preprocess_primary_frame_and_find_contours(initial_frame: np.ndarray):
    # Each image transform should be giving us back an np.ndarray of the same
    # shape and size.
    # out_frame = signal_transform.lip_removal(initial_frame)
    out_frame = downsample_image(initial_frame)
    out_frame = clahe_filter(out_frame)
    out_frame = percentile_threshold(out_frame)
    out_frame = invert_image(out_frame)
    out_frame = erosion_filter(out_frame)
    out_frame = remove_background(out_frame)
    out_frame = normalize_frame(out_frame)
    out_frame = unsharp_mask(out_frame)
    contours = find_contours(out_frame)
    return contours


def preprocess_color_channel(
        initial_frame: np.ndarray,
        color: str,
        alpha: float = 12,
        beta: int = 0,
):
    out_frame = downsample_image(initial_frame)
    out_frame = apply_brightness_contrast(
        out_frame,
        alpha,
        beta,
    )
    out_frame = colorize_frame(out_frame, color)
    return out_frame


def contour_filtration(contours):
    filtered_contours = cellular_highpass_filter(contours)
    return filtered_contours


def generate_frame_visualization(result: List[TiffPackage]):
    return generate_segmentation_visualization(
        filename='test',
        segmentation_results=result,
    )


def segmentation_pipeline(
        packaged_tiff: TiffPackage,
):
    primary_channel = copy.copy(packaged_tiff.img_array[0])
    auxiliary_channels = copy.copy(packaged_tiff.img_array[1:])
    # ------ SPLITTING OUT THE INPUT DATASTRUCTURE AND INITIAL PROCESSING ------
    for aux_channel_index in range(auxiliary_channels.shape[0]):
        packaged_tiff.cell_signal_auxiliary_frames.append([])
    for frame_index in range(primary_channel.shape[0]):
        # Handle the primary frame. We use the primary frame to color stuff in.
        primary_frame = primary_channel[frame_index]
        prime_contours = preprocess_primary_frame_and_find_contours(
            primary_frame
        )
        prime_contours = contour_filtration(prime_contours)
        packaged_tiff.processed_primary_frames.append(prime_contours)
        mask_frame = generate_mask(primary_frame, prime_contours)
        packaged_tiff.mask_frames.append(mask_frame)
        # Now we iterate over the other channels.
        aux_frame_container = []
        for aux_channel_index in range(auxiliary_channels.shape[0]):
            # Offset as we assume Channel Zero is our primary (typically grey)
            # frame.
            color = COLOR_LUT[packaged_tiff.channel_names[aux_channel_index + 1]]
            aux_frame = auxiliary_channels[aux_channel_index][frame_index]
            aux_processed_output = preprocess_color_channel(
                aux_frame,
                color,
            )
            aux_frame_container.append(aux_processed_output)
        # ------------ CORRELATING CELL CONTOURS TO FLUORESCENCE SIGNAL ------------
        for aux_channel_index in range(auxiliary_channels.shape[0]):
            aux_frame = auxiliary_channels[aux_channel_index][frame_index]
            correlated_cells = fluorescence_detection(
                primary_frame,
                aux_frame,
                prime_contours,
            )
            packaged_tiff.cell_signal_auxiliary_frames[aux_channel_index].append(correlated_cells)
        frame_stats = generate_arbitrary_stats(
            packaged_tiff.cell_signal_auxiliary_frames
        )
        # ----------------- LABELING FRAMES WITH DETECTED SIGNALS ------------------
        aux_labeled_frame = []
        for aux_channel_index in range(auxiliary_channels.shape[0]):
            aux_frame = auxiliary_channels[aux_channel_index][frame_index]
            out_label = label_cells(
                downsample_image(aux_frame),
                prime_contours,
                frame_stats[aux_channel_index][frame_index],
            )
            aux_labeled_frame.append(out_label)
        packaged_tiff.labeled_auxiliary_frames.append(aux_labeled_frame)
        # ----------------- COMBINING FRAMES FOR VISUALIZATION  ------------------
        out_frame = primary_frame
        for aux_channel_index in range(auxiliary_channels.shape[0]):
            aux_frame = auxiliary_channels[aux_channel_index][frame_index]
            out_frame = combine_frame(
                out_frame,
                aux_frame,
            )
        packaged_tiff.final_frames.append(out_frame)
    return packaged_tiff


def enqueue_segmentation(input_fp: str):
    # We should have a top level metadata xml file and then we have a directory
    # called raw_tiffs that has all of the stuff that we really need to work on.
    # We need to take the xml file and extract the channels, sizes, and
    # resolutions and use that to create a class object that can encapuslate the
    # logic related to segmenting tiffs of various dimensions and properties.
    global DEBUG
    metadata_fp = os.path.join(input_fp, 'metadata.xml')
    tree = ET.parse(metadata_fp)
    # This assumes that everyone has the same amount of channels.
    # If we get to the point where ND2 files have different channels WITHIN
    # themselves I'm throwing my computer into the Charles...
    channels = get_channel_data_from_xml_metadata(tree)
    packaged_tiffs = []
    tiff_path = os.path.join(input_fp, 'raw_tiffs')
    tiff_fp = glob.glob(tiff_path + '**/*.tiff')
    sorted_tiffs = sorted(tiff_fp, key=lambda x: int(Path(x).stem))
    for index, tiff_file in tqdm.tqdm(
            enumerate(sorted_tiffs),
            total=len(sorted_tiffs)
    ):
        array = ingress_tiff_file(tiff_file)
        if not array.shape[0]:
            continue
        wavelengths = [x[0] for x in channels[index]]
        channel_names = [x[1] for x in channels[index]]
        title = Path(tiff_file).stem
        inner_pack = TiffPackage(
            img_array=array,
            tiff_name=title,
            channel_names=channel_names,
            channel_wavelengths=wavelengths,
        )
        packaged_tiffs.append(inner_pack)
    # ---------------------------- PERFORM SEGMENTATION  -----------------------
    output_location = os.path.join(input_fp, 'segmentation_output')
    if not os.path.exists(output_location):
        os.mkdir(output_location)
    if DEBUG:
        segmentation_results = list(
            tqdm.tqdm(
                map(
                    segmentation_pipeline,
                    packaged_tiffs
                ),
                total=len(packaged_tiffs)
            )
        )
    else:
        with mp.get_context("spawn").Pool(processes=PROCESSES) as pool:
            segmentation_results = list(
                tqdm.tqdm(
                    pool.imap(segmentation_pipeline, packaged_tiffs),
                    total=len(packaged_tiffs)
                )
            )
    # ------------------- EXIT GRACEFULLY IF SEGMENTATION FAILED  --------------
    if not segmentation_results:
        return
    # ---------------- ENSURE OUTPUT DIRECTORY EXISTS AND WRITE OUT  -----------
    # TODO: ENSURE THAT THIS GUY IS SETUP RIGHT. IT SHOULD BE A LIST PER CHANNEL.
    for i, result in enumerate(tqdm.tqdm(segmentation_results)):
        for j, channel_result in enumerate(result.cell_signal_auxiliary_frames):
            frame_output = os.path.join(output_location, f'{i + 1}')
            channel_name = result.channel_names[j + 1]
            csv_location = os.path.join(frame_output, f'{i + 1}_{channel_name}.csv')
            if not os.path.exists(frame_output):
                os.mkdir(frame_output)
            with open(csv_location, 'a') as out_file:
                writer = csv.writer(out_file)
                for frame_result in channel_result:
                    writer.writerow([k.sum_signal for k in frame_result])
    if RENDER_VIDEOS:
        if DEBUG:
            c_results = list(
                tqdm.tqdm(
                    map(
                        generate_frame_visualization,
                        segmentation_results
                    ),
                    total=len(segmentation_results)
                )
            )
        else:
            with mp.get_context("spawn").Pool(processes=PROCESSES) as pool:
                c_results = list(
                    tqdm.tqdm(
                        pool.imap(
                            generate_frame_visualization,
                            segmentation_results,
                        ),
                        desc="Generating Visualizations...",
                        total=len(segmentation_results)
                    )
                )
        write_list = []
        # # iter = list(sigpro_utility.list_chunking(canvas_list, 20))
        # # file_list = []
        # # TODO: I need to balance moving things out of memory in the main python
        # #  function with the speed loss of hitting the disk for every file.
        # file_handles = list(
        #     tqdm.tqdm(
        #         map(
        #             write_out,
        #             write_list,
        #         ),
        #         desc="Writing Everything to disk...",
        #         total=len(input_frames)))
        # final_output = []
        # for file in file_handles:
        #     final_output.append(imageio.imread(file))
        #     os.remove(file)
        # imageio.mimsave(f'{output_directory}/{f_index}_video.gif', final_output)


def filter_inputs_based_on_channel(
        input_directory: str,
        filter_criteria: int = 2,
) -> bool:
    """

    Args:
        input_directory:
        filter_criteria:

    Returns:

    """
    metadata_filepath = os.path.join(input_directory, 'metadata.xml')
    tree = ET.parse(metadata_filepath)
    channels = get_channel_data_from_xml_metadata(tree)
    if len(channels[0]) != filter_criteria:
        return False
    return True


# ---------------------------- Application Commands ----------------------------
@app.command()
def convert_nd2_to_tiffs(
        input_directory: str = '/media/jackson/967e3f65-43ea-4c63-bcdd-e124725e6d63/microscopy_images/sam_2021',
        output_directory: str = '/media/jackson/967e3f65-43ea-4c63-bcdd-e124725e6d63/beholder_output',
        logging: bool = True,
):
    """

    Args:
        input_directory:
        output_directory:
        logging:

    Returns:

    """
    for dir_path in [input_directory, output_directory]:
        validate_dir_path(dir_path)
    selection_list = ['all']
    file_paths = glob.iglob(input_directory + '**/*.nd2', recursive=True)
    files_and_sizes = ((path, os.path.getsize(path)) for path in file_paths)
    sorted_files_with_size = sorted(files_and_sizes, key=operator.itemgetter(1))
    clean_filepaths = [file_path for file_path, _ in sorted_files_with_size]
    file_names = [Path(file_path).stem for file_path in clean_filepaths]
    for file_name in file_names:
        selection_list.append(file_name)
    terminal_menu = TerminalMenu(
        selection_list,
        multi_select=True,
        show_multi_select_hint=True,
    )
    menu_entry_indices = terminal_menu.show()
    if 'all' in terminal_menu.chosen_menu_entries:
        conversion_list = clean_filepaths
    else:
        out_list = []
        for index in menu_entry_indices:
            # Offsetting the 'All' that we started the list with.
            index = index - 1
            out_list.append(clean_filepaths[index])
        conversion_list = out_list
    # We have our selected input files and now we have to make sure that they
    # have a home...
    for input_fp in conversion_list:
        if logging:
            beholder_text(f'Converting {Path(input_fp).stem} to Tiff Files...')
        out_dir = os.path.join(output_directory, Path(input_fp).stem)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        nd2_convert(input_fp, out_dir)


@app.command()
def segmentation(
        input_directory: str = '/media/prime/beholder_output',
        render_videos: bool = True,
        logging: bool = True,
        filter_criteria: int = 3,
):
    """

    Args:
        input_directory:
        render_videos:
        logging:
        filter_criteria:

    Returns:

    """
    global RENDER_VIDEOS
    RENDER_VIDEOS = render_videos
    for dir_path in [input_directory]:
        validate_dir_path(dir_path)
    input_directories = []
    for x in os.scandir(input_directory):
        if x.is_dir:
            input_directories.append(x.path)
    files_and_sizes = ((
        path,
        sum([file.stat().st_size for file in Path(path).rglob('*')])) for path in input_directories)
    sorted_files_with_size = sorted(files_and_sizes, key=operator.itemgetter(1))
    input_directories = [file_path for file_path, _ in sorted_files_with_size]
    filter_fn = functools.partial(
        filter_inputs_based_on_channel,
        filter_criteria=filter_criteria,
    )
    input_directories = list((filter(filter_fn, input_directories)))
    display_directories = [Path(i).stem for i in input_directories]
    display_directories.insert(0, 'all')
    beholder_text('⬤ Please select input directories.')
    beholder_text(
        '-' * 88
    )
    terminal_menu = TerminalMenu(
        display_directories,
        multi_select=True,
        show_multi_select_hint=True,
    )
    menu_entry_indices = terminal_menu.show()
    if 'all' in terminal_menu.chosen_menu_entries:
        segmentation_list = input_directories
        segmentation_list.pop(0)
    else:
        out_list = []
        for index in menu_entry_indices:
            # Offsetting the 'All' that we started the list with.
            index = index - 1
            out_list.append(input_directories[index])
        segmentation_list = out_list
    # We have our selected input files and now we have to make sure that they
    # have a home...
    for input_directory in segmentation_list:
        if logging:
            beholder_text(
                f'⬤ Starting Segmentation Pipeline for '
                f'{Path(input_directory).stem}...'
            )
            beholder_text(
                '-' * 88
            )
        enqueue_segmentation(input_directory)
    typer.Exit()


@app.command()
def s3_sync(
        input_directory: str = '/media/prime/beholder_output',
        output_bucket: str = 'beholder-output',
):
    beholder_text(f'⬤ Syncing {input_directory} to AWS S3 Bucket {output_bucket}.')
    beholder_text('-' * 88)
    cmd = ['aws', 's3', 'sync', f'{input_directory}', f's3://{output_bucket}']
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True) as proc:
        for stdout_line in proc.stdout:
            sys.stdout.write(f'{stdout_line}\r')
            sys.stdout.flush()


if __name__ == "__main__":
    javabridge.start_vm(class_path=bf.JARS)
    mp.set_start_method("spawn")
    app()
