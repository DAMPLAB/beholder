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
import functools
import multiprocessing as mp
import operator
import os
import subprocess
import sys
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import (
    List,
)

import bioformats as bf
import javabridge
import typer
from rich.console import Console
from simple_term_menu import TerminalMenu

from beholder.pipelines import (
    enqueue_segmentation,
    enqueue_frame_stabilization,
)
from beholder.signal_processing.sigpro_utility import (
    nd2_convert,
    get_channel_data_from_xml_metadata,
)
from beholder.utils import (
    ConfigOptions,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")

console = Console()
app = typer.Typer()

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


# ------------------------ Terminal Rendering Commands -------------------------
def dataset_selection(
        input_directory: str,
        filter_criteria: int = None,
) -> List[str]:
    """
    
    Args:
        input_directory:
        filter_criteria:

    Returns:

    """
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
    if filter_criteria is not None:
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
    return segmentation_list


# -------------------------- Date Generation Commands --------------------------
@app.command()
def segmentation(
        input_directory: str = '/mnt/core2/beholder_output',
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
    ConfigOptions(render_videos=render_videos)
    segmentation_list = dataset_selection(
        input_directory=input_directory,
        filter_criteria=filter_criteria,
    )
    # We have our selected input files and now we have to make sure that they
    # have a home...
    for index, input_directory in enumerate(segmentation_list):
        if logging:
            beholder_text(
                f'⬤ Starting Segmentation Pipeline for '
                f'{Path(input_directory).stem}... ({index}/{len(segmentation_list)})'
            )
            beholder_text(
                '-' * 88
            )
        enqueue_segmentation(input_directory)
    typer.Exit()


@app.command()
def calculate_frame_drift(
        input_directory: str = '/mnt/core2/beholder_output',
        render_videos: bool = True,
        logging: bool = True,
        filter_criteria: int = 3,
):
    # We should use our standard tooling for determining the input directory.
    # Once we have our input directories, we feed them into our stabilization
    # function. Our stabilization function should output a JSON with a list
    # of xy transforms for each of the observation sets. We then use that
    # stabilization data during cell-flow, cell-tracking, and segmentation.
    stabilization_list = dataset_selection(
        input_directory=input_directory,
        filter_criteria=filter_criteria,
    )
    # We have our selected input files and now we have to make sure that they
    # have a home...
    for index, input_directory in enumerate(stabilization_list):
        if logging:
            beholder_text(
                f'⬤ Starting  Pipeline for '
                f'{Path(input_directory).stem}... ({index}/{len(stabilization_list)})'
            )
            beholder_text(
                '-' * 88
            )
        enqueue_frame_stabilization(input_directory)
    typer.Exit()


# ------------------------------ Utility Commands ------------------------------
@app.command()
def convert_nd2_to_tiffs(
        input_directory: str = '/media/prime/microscopy_images/sam_2021',
        output_directory: str = '/media/prime/beholder_output',
        filter_criteria: int = None,
        logging: bool = True,
):
    """

    Args:
        input_directory:
        output_directory:
        logging:

    Returns:

    """
    conversion_list = dataset_selection(
        input_directory=input_directory,
        filter_criteria=filter_criteria,
    )
    # We have our selected input files and now we have to make sure that they
    # have a home...
    for index, input_fp in enumerate(conversion_list):
        if logging:
            beholder_text(
                f'Converting {Path(input_fp).stem} to Tiff Files.. ({index}/{len(conversion_list)}).'
            )
        out_dir = os.path.join(output_directory, Path(input_fp).stem)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        nd2_convert(input_fp, out_dir)


@app.command()
def s3_sync_upload(
        input_directory: str = '/media/core2/beholder_output',
        output_bucket: str = 'beholder-output',
        results_only: bool = True,
):
    """

    Args:
        input_directory:
        output_bucket:
        results_only:

    Returns:

    """
    beholder_text(f'⬤ Syncing {input_directory} to AWS S3 Bucket {output_bucket}.')
    beholder_text('-' * 88)
    cmd = ['aws', 's3', 'sync', '--size-only', f'{input_directory}', f's3://{output_bucket}']
    if results_only:
        cmd.append('--exclude')
        cmd.append('*.tiff')
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True) as proc:
        for stdout_line in proc.stdout:
            sys.stdout.write(f'{stdout_line}\r')
            sys.stdout.flush()


@app.command()
def s3_sync_download(
        output_directory: str = '/mnt/core2/beholder_test',
        input_bucket: str = 'beholder-output',
        results_only: bool = True,
):
    """

    Args:
        output_directory:
        input_bucket:
        results_only:

    Returns:

    """
    beholder_text(f'⬤ Downloading AWS S3 Bucket {input_bucket} to local directory {output_directory}...')
    beholder_text('-' * 88)
    cmd = ['aws', 's3', 'sync', f's3://{input_bucket}', f'{output_directory}', '--exclude', '*.tiff']
    if results_only:
        cmd.append('--exclude')
        cmd.append('*.tiff')
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True) as proc:
        for stdout_line in proc.stdout:
            sys.stdout.write(f'{stdout_line}\r')
            sys.stdout.flush()


if __name__ == "__main__":
    javabridge.start_vm(class_path=bf.JARS)
    mp.set_start_method("spawn")
    app()
