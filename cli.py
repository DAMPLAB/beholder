'''
--------------------------------------------------------------------------------
Written by W.R. Jackson <wrjackso@bu.edu>, DAMP Lab 2021
--------------------------------------------------------------------------------
'''
import functools
import json
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
    Union,
)

import typer
from rich.console import Console
from simple_term_menu import TerminalMenu

from beholder.pipelines import (
    enqueue_segmentation,
    enqueue_frame_stabilization,
    enqueue_brute_force_conversion,
    enqueue_nd2_conversion,
    enqueue_panel_detection,
)
from beholder.signal_processing.sigpro_utility import (
    get_channel_and_wl_data_from_xml_metadata,
    get_channel_data_from_xml_metadata,
)
from beholder.utils import (
    BLogger,
    ConfigOptions,
    get_color_keys,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")

console = Console()
app = typer.Typer()
log = BLogger()

# Jackson is lazy variables
ND2_LOC = '/mnt/core1/3-Microscope_Images/Batch1'
OUT_LOC = '/mnt/core1/beholder_output'

# ----------------------- Command Line Utility Functions -----------------------


def validate_dir_path(input_fp: str):
    if not os.path.isdir(input_fp):
        raise RuntimeError(
            f'Unable to locate {input_fp}. Please check input and try again.'
        )


def filter_inputs_based_on_number_of_channel(
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
    channels = get_channel_and_wl_data_from_xml_metadata(tree)
    if len(channels[0]) != filter_criteria:
        return False
    return True


def filter_inputs_based_on_name_of_channel(
        input_directory: str,
        filter_criteria: str = 'DAPI1',
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
    return filter_criteria in channels


def filter_inputs_based_on_name_of_dataset(
        input_directory: str,
        filter_criteria: str = '',
) -> bool:
    """

    Args:
        input_directory:
        filter_criteria:

    Returns:

    """
    dataset_name = Path(input_directory).stem
    return filter_criteria in dataset_name


# ------------------------ Terminal Rendering Commands -------------------------
def dataset_selection(
        input_directory: str,
        filter_criteria: Union[str, int] = None,
        sort_criteria: str = 'size',
) -> List[str]:
    """
    
    Args:
        input_directory:
        filter_criteria:
        sort_criteria:

    Returns:

    """
    for dir_path in [input_directory]:
        validate_dir_path(dir_path)
    input_directories = []
    for x in os.scandir(input_directory):
        if x.is_dir:
            input_directories.append(x.path)
    if sort_criteria == 'size':
        files_and_sizes = ((
            path,
            sum([file.stat().st_size for file in Path(path).rglob('*')])) for path in input_directories)
        sorted_files_with_size = sorted(files_and_sizes, key=operator.itemgetter(1))
        input_directories = [file_path for file_path, _ in sorted_files_with_size]
    if sort_criteria == 'alpha':
        input_directories = sorted(input_directories)
    if filter_criteria is not None:
        filter_fn = None
        # We are assuming that we are filtering based on the number of channels.
        if type(filter_criteria) == int:
            filter_fn = functools.partial(
                filter_inputs_based_on_number_of_channel,
                filter_criteria=filter_criteria,
            )
        # We want to be able to filter on strings so I can get what I want
        # without having to search for it.
        if type(filter_criteria) == str:
            # Channel Name Filtration
            if filter_criteria in get_color_keys():
                filter_fn = functools.partial(
                    filter_inputs_based_on_name_of_channel,
                    filter_criteria=filter_criteria,
                )
            # Dataset Observation Filtration
            else:
                filter_fn = functools.partial(
                    filter_inputs_based_on_name_of_dataset,
                    filter_criteria=filter_criteria,
                )
        if filter_fn is None:
            raise RuntimeError(
                'Filter fallthrough. Investigate filter calling format.'
            )
        input_directories = list((filter(filter_fn, input_directories)))
    display_directories = [Path(i).stem for i in input_directories]
    display_directories.insert(0, 'all')
    log.info('⬤ Please select input directories.')
    log.info(
        '-' * 88
    )
    terminal_menu = TerminalMenu(
        display_directories,
        multi_select=True,
        show_multi_select_hint=True,
    )
    menu_entry_indices = terminal_menu.show()
    if terminal_menu.chosen_menu_entries is None:
        log.info('Exit Recognized. Have a nice day!')
        exit(0)
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


@app.command()
def segmentation(
        input_directory: str = '/mnt/core2/beholder_output',
        render_videos: bool = True,
        logging: bool = True,
        filter_criteria=None,
        runlist: str = None,
        panel_selection: List[int] = None,
):
    """

    Args:
        input_directory:
        render_videos:
        logging:
        filter_criteria:
        runlist:
        panel_selection:

    Returns:

    """
    ConfigOptions(render_videos=render_videos)
    if runlist is None:
        segmentation_list = dataset_selection(
            input_directory=input_directory,
            filter_criteria=filter_criteria,
        )
    else:
        segmentation_list = runlist_validation_and_parsing(
            input_directory=input_directory,
            runlist_fp=runlist,
            files=False,
        )
    # We have our selected input files and now we have to make sure that they
    # have a home...
    if panel_selection is not None:
        temp_list = []
        # We assume that the selection is a list of integer values that correspond to the index of the desired panel.
        # e.g, if just the first panel is desired we should pass in [0], or if 2 and 7 are desired it should [2, 7]
        for selection in panel_selection:
            temp_list.append(segmentation_list[selection])
        segmentation_list = temp_list
    for index, input_directory in enumerate(segmentation_list):
        if logging:
            log.info(
                f'⬤ Starting Segmentation Pipeline for '
                f'{Path(input_directory).stem}... ({index}/{len(segmentation_list)})'
            )
            log.info(
                '-' * 88
            )
        enqueue_segmentation(input_directory)
    typer.Exit()


def runlist_validation_and_parsing(
        input_directory: str,
        runlist_fp: str = "example_runlist.json",
        files: bool = True,
) -> List[str]:
    """

    Args:
        input_directory
        runlist_fp
        files:

    Returns:

    """
    if not os.path.isfile(runlist_fp):
        raise RuntimeError('Unable to locate input runlist, please investigate.')
    with open(runlist_fp, 'r') as input_file:
        runlist = json.load(input_file)
        input_directories = runlist['input_datasets']
        if files:
            input_directories = [i + ".nd2" for i in input_directories]
        input_directories = [os.path.join(input_directory, i) for i in input_directories]
        removal_list = []
        for i in input_directories:
            if not os.path.isfile(i) and not os.path.isdir(i):
                log.info(f'Unable to locate {i}, skipping...')
                removal_list.append(i)
        for bad_dir in removal_list:
            input_directories.remove(bad_dir)
        return input_directories


# -------------------------- Date Generation Commands --------------------------
@app.command()
def check_panel_detection(
        input_directory: str = ND2_LOC,
        filter_criteria=None,
        runlist: str = None,
):
    """

    Args:
        input_directory:
        filter_criteria:
        runlist:

    Returns:

    """
    if type(filter_criteria) == str and filter_criteria in get_color_keys():
        raise RuntimeError(
            'Cannot do channel filtration on dataset prior to generation of a '
            'metadata file.'
        )
    if runlist is None:
        conversion_list = dataset_selection(
            input_directory=input_directory,
            filter_criteria=filter_criteria,
            sort_criteria='alpha',
        )
    else:
        conversion_list = runlist_validation_and_parsing(
            input_directory=input_directory,
            runlist_fp=runlist,
            files=True,
        )
    log.debug(f'Conversion list: {conversion_list}')
    log.change_logging_level('debug')
    enqueue_panel_detection(conversion_list)


@app.command()
def calculate_frame_drift(
        input_directory: str = '/mnt/core2/beholder_output',
        render_videos: bool = True,
        logging: bool = True,
        filter_criteria=3,
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
            log.info(
                f'⬤ Starting Frame-Shift Calculation Pipeline for '
                f'{Path(input_directory).stem}... ({index}/{len(stabilization_list)})'
            )
            log.info(
                '-' * 88
            )
        enqueue_frame_stabilization(input_directory)
    typer.Exit()


# ------------------------------ Utility Commands ------------------------------
@app.command()
def convert_nd2_to_tiffs(
        input_directory: str = ND2_LOC,
        output_directory: str = OUT_LOC,
        filter_criteria=None,
        runlist: str = None,
):
    """

    Args:
        input_directory:
        output_directory:
        filter_criteria:
        runlist:

    Returns:

    """
    if type(filter_criteria) == str and filter_criteria in get_color_keys():
        raise RuntimeError(
            'Cannot do channel filtration on dataset prior to generation of a '
            'metadata file.'
        )
    if runlist is None:
        conversion_list = dataset_selection(
            input_directory=input_directory,
            filter_criteria=filter_criteria,
            sort_criteria='alpha',
        )
    else:
        conversion_list = runlist_validation_and_parsing(
            input_directory=input_directory,
            runlist_fp=runlist,
            files=True,
        )
    log.debug(f'Conversion list: {conversion_list}')
    enqueue_nd2_conversion(conversion_list, output_directory)


@app.command()
def convert_corrupted_nd2_to_tiffs(
        input_directory: str = '/mnt/core2/microscopy_images/sam_2021',
        output_directory: str = '/mnt/core1/beholder_output',
        runlist: str = None,
):
    """

    Args:
        input_directory:
        output_directory:
        filter_criteria:
        runlist:

    Returns:

    """
    conversion_list = runlist_validation_and_parsing(
        input_directory=input_directory,
        runlist_fp=runlist,
        files=True,
    )
    log.debug(f'Conversion list: {conversion_list}')
    enqueue_brute_force_conversion(
        conversion_list=conversion_list,
        output_directory=output_directory,
        runlist_fp=runlist,
    )


@app.command()
def s3_sync_upload(
        input_directory: str = '/mnt/core2/beholder_output',
        output_bucket: str = 'beholder-output',
        results_only: bool = False,
        filter_criteria=None,
        runlist: str = None,
):
    """

    Args:
        input_directory:
        output_bucket:
        results_only:
        filter_criteria

    Returns:

    """
    if runlist is None:
        upload_list = dataset_selection(
            input_directory=input_directory,
            filter_criteria=filter_criteria,
        )
    else:
        upload_list = runlist_validation_and_parsing(
            input_directory=input_directory,
            runlist_fp=runlist,
            files=False,
        )
    log.info(f'⬤ Syncing {input_directory} to AWS S3 Bucket {output_bucket}.')
    log.info('-' * 88)
    for upload_dir in upload_list:
        upload_suffix = Path(upload_dir).stem
        cmd = ['aws', 's3', 'sync', '--size-only', f'{upload_dir}', f's3://{output_bucket}/{upload_suffix}']
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
    log.info(f'⬤ Downloading AWS S3 Bucket {input_bucket} to local directory {output_directory}...')
    log.info('-' * 88)
    cmd = ['aws', 's3', 'sync', f's3://{input_bucket}', f'{output_directory}', '--exclude', '*.tiff']
    if results_only:
        cmd.append('--exclude')
        cmd.append('*.tiff')
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True) as proc:
        for stdout_line in proc.stdout:
            sys.stdout.write(f'{stdout_line}\r')
            sys.stdout.flush()


# ------------------------------ God Command -----------------------------------
@app.command()
def beholder(
        runlist: str,
        nd2_directory: str = '/mnt/core1/3-Microscope_Images/Batch2',
        output_directory: str = '/mnt/core1/beholder_output',
        filter_criteria=None,
):
    # We just the pipeline in it's entirety, piping the arguments throughout
    # the entirety of the program.
    log.info('Beholder START.')
    if not os.path.isfile(runlist):
        raise RuntimeError(f'Cannot find runlist at {runlist}')
    # Extract our stages.
    with open(runlist, 'r') as input_file:
        runlist_json = json.load(input_file)
        stages = runlist_json['stages']
    for stage in stages:
        log.info(f'Starting Stage: {stage}...')
        if stage == "convert_nd2_to_gif":
            convert_nd2_to_tiffs(
                input_directory=nd2_directory,
                output_directory=output_directory,
                filter_criteria=filter_criteria,
                runlist=runlist,
            )
        if stage == "convert_corrupted_nd2_to_tiffs":
            convert_corrupted_nd2_to_tiffs(
                input_directory=nd2_directory,
                output_directory=output_directory,
                runlist=runlist,
            )
        if stage == "segmentation":
            segmentation(
                input_directory=output_directory,
                runlist=runlist,
            )
        if stage == "s3_sync_upload":
            s3_sync_upload(
                input_directory=output_directory,
                runlist=runlist,
            )
        if stage == "s3_sync_download":
            s3_sync_download(
                output_directory=output_directory,
            )
        log.info(f'Finishing Stage: {stage}...')


if __name__ == "__main__":
    mp.set_start_method("spawn")
    app()
