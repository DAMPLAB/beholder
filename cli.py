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
    enqueue_rpu_calculation,
    enqueue_lf_analysis,
    enqueue_figure_generation,
    enqueue_autofluorescence_calculation,
    enqueue_panel_based_gif_generation,
    enqueue_long_analysis,
    enqueue_wide_analysis,
    enqueue_porcelain_conversion,
    enqueue_dataset_split,
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
        input_directory: str = None,
        render_videos: bool = True,
        logging: bool = True,
        filter_criteria=None,
        runlist: str = None,
        do_segment: bool = True,
        do_defocus: bool = True,
        do_render_videos: bool = True,
        panel_selection: List[int] = None,
):
    """

    Args:
        input_directory:
        render_videos:
        logging:
        filter_criteria:
        runlist:
        do_segment:
        do_defocus:
        do_render_videos:
        panel_selection:

    Returns:

    """
    ConfigOptions(render_videos=render_videos)
    if input_directory is None:
        input_directory = ConfigOptions().output_location
    if runlist is None:
        segmentation_list = dataset_selection(
            input_directory=input_directory,
            filter_criteria=filter_criteria,
            sort_criteria='alpha',
        )
    else:
        segmentation_list = runlist_validation_and_parsing(
            input_directory=input_directory,
            runlist_fp=runlist,
            files=False,
        )
    # We have our selected input files and now we have to make sure that they
    # have a home...
    # if panel_selection is not None:
    #     temp_list = []
    #     # We assume that the selection is a list of integer values that correspond to the index of the desired panel.
    #     # e.g, if just the first panel is desired we should pass in [0], or if 2 and 7 are desired it should [2, 7]
    #     for selection in panel_selection:
    #         temp_list.append(segmentation_list[selection])
    #     segmentation_list = temp_list
    for index, input_directory in enumerate(segmentation_list):
        if logging:
            log.info(
                f'⬤ Starting Segmentation Pipeline for '
                f'{Path(input_directory).stem}... ({index}/{len(segmentation_list)})'
            )
            log.info(
                '-' * 88
            )
        enqueue_segmentation(
            input_fp=input_directory,
            do_segment=do_segment,
            do_defocus=do_defocus,
            do_render_videos=do_render_videos,
        )
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


def fp_expansion(
        input_directory: str,
        filename: str,
) -> str:
    """

    Args:
        input_directory:
        filename:

    Returns:

    """
    out_fp = os.path.join(input_directory, filename)
    if not os.path.exists(out_fp):
        raise RuntimeError(f'Failed to find passed in directory {out_fp}. Please Investigate.')
    return out_fp


def runlist_check(
        input_directory: str,
        runlist_fp: str = "example_runlist.json",
        files: bool = True,
) -> bool:
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
                log.warning(f'Missing the following dataset: {i}')
                removal_list.append(i)
        if removal_list:
            return False
        return True


# -------------------------- Date Generation Commands --------------------------
@app.command()
def check_panel_detection(
        input_directory: str = None,
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
    ConfigOptions()
    if input_directory is None:
        input_directory = ConfigOptions().nd2_location
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
def calculate_rpu_calibration_value(
        input_directory: str = None,
        rpu_fp: str = None,
        calibration_fp: str = None,
        filter_criteria=None,
):
    """

    Args:
        input_directory:
        filter_criteria:
        runlist:

    Returns:

    """
    ConfigOptions()
    if input_directory is None:
        input_directory = ConfigOptions().output_location
    if rpu_fp is None:
        log.info('Please select the RPU Calibration Dataset.')
        selection_list = dataset_selection(
            input_directory=input_directory,
            filter_criteria=filter_criteria,
            sort_criteria='alpha',
        )
        if len(selection_list) > 1:
            raise RuntimeError('RPU Calculation presupposes a singular directory.')
        rpu_input_fp = selection_list[0]
    else:
        dataset_fp = fp_expansion(input_directory, rpu_fp)
    if calibration_fp is None:
        log.info('Please select the Autofluorescence Calibration Dataset.')
        selection_list = dataset_selection(
            input_directory=input_directory,
            filter_criteria=filter_criteria,
            sort_criteria='alpha',
        )
        if len(selection_list) > 1:
            raise RuntimeError('Autofluorescence Calculation presupposes a singular directory.')
        autofluorescence_fp = selection_list[0]
    else:
        autofluorescence_fp = fp_expansion(input_directory, calibration_fp)
    log.debug(f'Selection for RPU Calculation: {dataset_fp}')
    log.debug(f'Selection for Autofluorescence Calculation: {autofluorescence_fp}')
    enqueue_rpu_calculation(
        rpu_input_fp=dataset_fp,
        autofluorescence_input_fp=autofluorescence_fp,
    )


@app.command()
def calculate_autofluorescence_calibration_value(
        input_directory: str = None,
        selected_dataset: str = None,
        filter_criteria=None,
):
    """

    Args:
        input_directory:
        filter_criteria:
        runlist:

    Returns:

    """
    ConfigOptions()
    if input_directory is None:
        input_directory = ConfigOptions().output_location
    if selected_dataset is None:
        selection_list = dataset_selection(
            input_directory=input_directory,
            filter_criteria=filter_criteria,
            sort_criteria='alpha',
        )
        if len(selection_list) > 1:
            raise RuntimeError('RPU Calculation presupposes a singular directory.')
        dataset_fp = selection_list[0]
    else:
        # TODO: Path Join and Validation for this dataset right here.
        dataset_fp = fp_expansion(input_directory, selected_dataset)
    log.change_logging_level('debug')
    log.debug(f'Selection for Autofluorescence Calculation: {dataset_fp}')
    enqueue_autofluorescence_calculation(dataset_fp)


@app.command()
def split_input_dataset(
        input_directory: str = None,
        output_directory: str = None,
        runlist_fp: str = None,
):
    """

    Args:
        input_directory:
        output_directory:
        selected_dataset:
        runlist_fp:
        panel_distribution:

    Returns:

    """
    ConfigOptions()
    if input_directory is None or output_directory is None:
        input_directory = ConfigOptions().nd2_location
        output_directory = ConfigOptions().output_location
    conversion_list = runlist_validation_and_parsing(
        input_directory=input_directory,
        runlist_fp=runlist_fp,
        files=True,
    )
    log.debug(f'Conversion list: {conversion_list}')
    enqueue_dataset_split(
        input_directory=input_directory,
        output_directory=output_directory,
        runlist_fp=runlist_fp,
        conversion_list=conversion_list,
    )



@app.command()
def calculate_frame_drift(
        input_directory: str = None,
        render_videos: bool = True,
        logging: bool = True,
        filter_criteria=3,
):
    # We should use our standard tooling for determining the input directory.
    # Once we have our input directories, we feed them into our stabilization
    # function. Our stabilization function should output a JSON with a list
    # of xy transforms for each of the observation sets. We then use that
    # stabilization data during cell-flow, cell-tracking, and segmentation.
    ConfigOptions()
    if input_directory is None:
        input_directory = ConfigOptions().output_location
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


@app.command()
def perform_lf_analysis(
        runlist: str,
        calibration_rpu_dataset: str,
        input_directory: str = None,
):
    """

    Args:
        runlist:
        calibration_rpu_dataset:
        input_directory:

    Returns:

    """
    ConfigOptions()
    if input_directory is None:
        input_directory = ConfigOptions().output_location
    bound_datasets = runlist_validation_and_parsing(
        input_directory=input_directory,
        runlist_fp=runlist,
        files=False,
    )
    calibration_rpu_dataset_fp = os.path.join(
        input_directory,
        calibration_rpu_dataset,
        'rpu_correlation_value.csv',
    )
    enqueue_lf_analysis(
        input_datasets=bound_datasets,
        calibration_rpu_dataset_fp=calibration_rpu_dataset_fp,
        runlist_fp=runlist,
    )


@app.command()
def perform_long_analysis(
        runlist: str,
        input_directory: str = None,
):
    """

    Args:
        runlist:
        calibration_rpu_dataset:
        input_directory:

    Returns:

    """
    ConfigOptions()
    if input_directory is None:
        input_directory = ConfigOptions().output_location
    bound_datasets = runlist_validation_and_parsing(
        input_directory=input_directory,
        runlist_fp=runlist,
        files=False,
    )
    enqueue_long_analysis(
        input_datasets=bound_datasets,
        runlist_fp=runlist,
    )


@app.command()
def perform_wide_analysis(
        runlist: str,
        calibration_rpu_dataset: str,
        calibration_af_dataset: str,
        input_directory: str = None,
):
    """

    Args:
        runlist:
        calibration_rpu_dataset:
        input_directory:

    Returns:

    """
    ConfigOptions()
    if input_directory is None:
        input_directory = ConfigOptions().output_location
    bound_datasets = runlist_validation_and_parsing(
        input_directory=input_directory,
        runlist_fp=runlist,
        files=False,
    )
    calibration_rpu_dataset_fp = os.path.join(
        input_directory,
        calibration_rpu_dataset,
        'rpu_correlation_value.csv',
    )
    af_dataset_fp = os.path.join(
        input_directory,
        calibration_af_dataset,
        'autofluorescence_correlation_value.csv',
    )
    enqueue_wide_analysis(
        input_datasets=bound_datasets,
        calibration_rpu_dataset_fp=calibration_rpu_dataset_fp,
        calibration_autofluoresence_dataset_fp=af_dataset_fp,
        runlist_fp=runlist,
    )


@app.command()
def generate_panel_based_gifs(
        runlist: str,
        input_directory: str = None,
        alpha: int = 12,
        beta: int = 0,
):
    """

    Args:
        runlist:
        input_directory:
        alpha:
        beta:

    Returns:

    """
    ConfigOptions()
    if input_directory is None:
        input_directory = ConfigOptions().output_location
    bound_datasets = runlist_validation_and_parsing(
        input_directory=input_directory,
        runlist_fp=runlist,
        files=False,
    )
    enqueue_panel_based_gif_generation(
        input_datasets=bound_datasets,
        runlist_fp=runlist,
        alpha=alpha,
        beta=beta,
    )


@app.command()
def generate_figures(
        input_directory: str,
        figure_type: str = 'longform_analysis',
):
    """

    Args:
        input_directory:
        figure_type:

    Returns:

    """
    ConfigOptions()
    if input_directory is None:
        input_directory = ConfigOptions().output_location
    enqueue_figure_generation(
        input_fp=input_directory,
        figure_type=figure_type,
    )


# ------------------------------ Utility Commands ------------------------------
@app.command()
def convert_nd2_to_tiffs(
        input_directory: str = None,
        output_directory: str = None,
        filter_criteria=None,
        runlist: str = None,
        force_reconversion: bool = True,
):
    """

    Args:
        input_directory:
        output_directory:
        filter_criteria:
        runlist:
        force_reconversion:

    Returns:

    """
    ConfigOptions()
    if input_directory is None:
        input_directory = ConfigOptions().nd2_location
        output_directory = ConfigOptions().output_location
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
    enqueue_nd2_conversion(
        conversion_list=conversion_list,
        output_directory=output_directory,
        force_reconversion=force_reconversion,
    )


@app.command()
def convert_corrupted_nd2_to_tiffs(
        input_directory: str = None,
        output_directory: str = None,
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
    if input_directory is None:
        input_directory = ConfigOptions().nd2_location
        output_directory = ConfigOptions().output_location
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
        input_directory: str = None,
        s3_bucket: str = None,
        results_only: bool = False,
        filter_criteria=None,
        runlist: str = None,
):
    """

    Args:
        input_directory:
        s3_bucket:
        results_only:
        filter_criteria

    Returns:

    """
    if input_directory is None:
        input_directory = ConfigOptions().nd2_location
        s3_bucket = ConfigOptions().s3_bucket
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
    log.info(f'⬤ Syncing {input_directory} to AWS S3 Bucket {s3_bucket}.')
    log.info('-' * 88)
    for upload_dir in upload_list:
        upload_suffix = Path(upload_dir).stem
        cmd = ['aws', 's3', 'sync', '--size-only', f'{upload_dir}', f's3://{s3_bucket}/{upload_suffix}']
        if results_only:
            cmd.append('--exclude')
            cmd.append('*.tiff')
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True) as proc:
            for stdout_line in proc.stdout:
                sys.stdout.write(f'{stdout_line}\r')
                sys.stdout.flush()


@app.command()
def s3_sync_download(
        output_directory: str = None,
        s3_bucket: str = None,
        results_only: bool = True,
):
    """

    Args:
        output_directory:
        s3_bucket:
        results_only:

    Returns:

    """
    if output_directory is None:
        output_directory = ConfigOptions().output_location
        s3_bucket = ConfigOptions().s3_bucket
    log.info(f'⬤ Downloading AWS S3 Bucket {s3_bucket} to local directory {output_directory}...')
    log.info('-' * 88)
    cmd = ['aws', 's3', 'sync', f's3://{s3_bucket}', f'{output_directory}', '--exclude', '*.tiff']
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
        nd2_directory: str = None,
        output_directory: str = None,
        filter_criteria=None,
):
    # We just the pipeline in it's entirety, piping the arguments throughout
    # the entirety of the program.
    ConfigOptions()
    if nd2_directory is None:
        nd2_directory = ConfigOptions().nd2_location
        output_directory = ConfigOptions().output_location
    log.info('Beholder START.')
    if not os.path.isfile(runlist):
        raise RuntimeError(f'Cannot find runlist at {runlist}')
    # Extract our stages.
    with open(runlist, 'r') as input_file:
        runlist_json = json.load(input_file)
        stages = runlist_json['stages']
        stage_settings = runlist_json['settings']
    for stage in stages:
        log.info(f'Starting Stage: {stage}...')
        if stage == "convert_nd2_to_tiffs":
            if "convert_nd2_to_tiffs" in stage_settings:
                convert_nd2_to_tiffs(
                    input_directory=nd2_directory,
                    output_directory=output_directory,
                    filter_criteria=filter_criteria,
                    runlist=runlist,
                    **stage_settings['convert_nd2_to_tiffs'],
                )
            else:
                convert_nd2_to_tiffs(
                    input_directory=nd2_directory,
                    output_directory=output_directory,
                    filter_criteria=filter_criteria,
                    runlist=runlist,
                )
        elif stage == "convert_corrupted_nd2_to_tiffs":
            if "convert_corrupted_nd2_to_tiffs" in stage_settings:
                convert_corrupted_nd2_to_tiffs(
                    input_directory=nd2_directory,
                    output_directory=output_directory,
                    runlist=runlist,
                    **stage_settings["convert_corrupted_nd2_to_tiffs"]
                )
            else:
                convert_corrupted_nd2_to_tiffs(
                    input_directory=nd2_directory,
                    output_directory=output_directory,
                    runlist=runlist,
                )
        elif stage == "segmentation":
            if "segmentation" in stage_settings:
                segmentation(
                    input_directory=output_directory,
                    runlist=runlist,
                    **stage_settings['segmentation']
                )
            else:
                segmentation(
                    input_directory=output_directory,
                    runlist=runlist,
                )
        elif stage == "s3_sync_upload":
            if "s3_sync_upload" in stage_settings:
                s3_sync_upload(
                    input_directory=output_directory,
                    runlist=runlist,
                    **stage_settings['s3_sync_upload']
                )
            else:
                s3_sync_upload(
                    input_directory=output_directory,
                    runlist=runlist,
                )
        elif stage == "calculate_autofluorescence_calibration_value":
            if "calculate_autofluorescence_calibration_value" in stage_settings:
                calculate_autofluorescence_calibration_value(
                    input_directory=output_directory,
                    **stage_settings['calculate_autofluorescence_calibration_value']
                )
            else:
                calculate_autofluorescence_calibration_value(
                    input_directory=output_directory,
                )
        elif stage == "calculate_rpu_calibration_value":
            if "calculate_rpu_calibration_value" in stage_settings:
                calculate_rpu_calibration_value(
                    input_directory=output_directory,
                    **stage_settings['calculate_rpu_calibration_value']
                )
            else:
                calculate_autofluorescence_calibration_value(
                    input_directory=output_directory,
                )
        elif stage == "split_input_dataset":
            if "split_input_dataset" in stage_settings:
                split_input_dataset(
                    input_directory=output_directory,
                    runlist_fp=runlist,
                )
            else:
                raise RuntimeError(
                    'Unable to perform dataset split without panel distribution'
                    'being annotated within runlist. Please investigate.'
                )
        elif stage == "perform_long_analysis":
            if "perform_long_analysis" in stage_settings:
                perform_long_analysis(
                    input_directory=output_directory,
                    **stage_settings['perform_long_analysis']
                )
            else:
                perform_long_analysis(
                    runlist=runlist,
                    input_directory=output_directory,
                )
        elif stage == "perform_wide_analysis":
            if "perform_wide_analysis" in stage_settings:
                perform_wide_analysis(
                    input_directory=output_directory,
                    runlist=runlist,
                    **stage_settings['perform_wide_analysis']
                )
            else:
                perform_wide_analysis(
                    runlist=runlist,
                    input_directory=output_directory,
                )
        elif stage == "s3_sync_download":
            s3_sync_download(
                output_directory=output_directory,
            )
        elif stage == 'run_lf_analysis':
            if "run_lf_analysis" in stage_settings:
                perform_lf_analysis(
                    input_directory=output_directory,
                    runlist=runlist,
                    **stage_settings['run_lf_analysis']
                )
            else:
                perform_lf_analysis(
                    input_directory=output_directory,
                    runlist=runlist,
                )
        elif stage == 'generate_panel_based_gifs':
            if "generate_panel_based_gifs" in stage_settings:
                generate_panel_based_gifs(
                    input_directory=output_directory,
                    runlist=runlist,
                    **stage_settings['run_lf_analysis']
                )
            else:
                generate_panel_based_gifs(
                    input_directory=output_directory,
                    runlist=runlist,
                )
        else:
            log.warning(f'Stage {stage} not recognized as valid pipeline stage.')
        log.info(f'Finishing Stage: {stage}...')


@app.command()
def batchholder(
        batchlist_fp: str,
        nd2_directory: str = None,
):
    """

    Args:
        batchlist_fp:
        nd2_directory:
        output_directory:
        filter_criteria:

    Returns:

    """
    # We just the pipeline in it's entirety, piping the arguments throughout
    # the entirety of the program.
    ConfigOptions()
    if nd2_directory is None:
        nd2_directory = ConfigOptions().nd2_location
        output_directory = ConfigOptions().output_location
    log.info('Batchholder START.')
    if not os.path.isfile(batchlist_fp):
        raise RuntimeError(f'Cannot find batch runlist at {batchlist_fp}')
    # Extract our stages.
    with open(batchlist_fp, 'r') as input_file:
        batch_list = json.load(input_file)
        runlist_abs_path = batch_list['absolute_path']
        runlists = batch_list['runlists']
    runlist_filepaths = list(map(lambda x: os.path.join(runlist_abs_path, f'{x}.json'), runlists))
    # We assume that we're doing the ND2 conversion here. I'll remove this at
    # a later date if we stick with the longform batch model.
    for runlist in runlist_filepaths:
        if not os.path.isfile(runlist):
            raise RuntimeError(f'Cannot find {runlist}. Please investigate.')
        # Then we want to make sure that all of our datasets exist where we say
        # they should be, because I'm not constantly dicking with this.
        if not runlist_check(
                input_directory=nd2_directory,
                runlist_fp=runlist,
                files=True,
        ):
            raise RuntimeError(
                f'Unable to find datasets for {runlist}. Please investigate.'
            )
    enqueue_porcelain_conversion(
        nd2_directory=nd2_directory,
        output_directory=output_directory,
        batchlist_fp=batchlist_fp,
    )
    for runlist in list(runlist_filepaths):
        beholder(runlist=runlist)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    app()
