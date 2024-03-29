import datetime
import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import (
    Dict,
    List,
)
import json
from rich.console import Console

from beholder.utils.gof import SingletonBaseClass


@dataclass
class ConfigOptions(metaclass=SingletonBaseClass):
    color_lut: Dict[str, str] = None

    render_videos: bool = True
    max_processes: int = mp.cpu_count() - 1
    single_thread_debug: bool = False
    visualization_debug: bool = False
    test_write: bool = True
    prior_runlist_fp: str = None

    nd2_location: str = None
    output_location: str = None
    s3_bucket: str = None

    analysis_location: str = None

    def __post_init__(self):
        if self.color_lut is None:
            self.color_lut = {
                'PhC': 'grey',
                'm-Cherry': 'red',
                'DAPI1': 'green',
                'YFP': 'yellow',
                'GFP': 'green',
            }
        if os.path.exists('config.json'):
            with open('config.json') as input_config:
                in_json = json.load(input_config)
                if 'nd2_location' in in_json:
                    self.nd2_location = in_json['nd2_location']
                if "output_location" in in_json:
                    self.output_location = in_json['output_location']
                if "s3_bucket" in in_json:
                    self.output_location = in_json['s3_bucket']


def get_color_keys() -> List[str]:
    return list(ConfigOptions().color_lut.keys())


def do_render_videos() -> bool:
    return ConfigOptions().render_videos


def get_max_processes() -> int:
    return ConfigOptions().max_processes


def do_single_threaded() -> bool:
    return ConfigOptions().single_thread_debug


def do_visualization_debug() -> bool:
    return ConfigOptions().visualization_debug


def convert_channel_name_to_color(channel_name: str) -> str:
    return ConfigOptions().color_lut[channel_name]


def do_test_write() -> bool:
    return ConfigOptions().test_write


def get_nd2_file_location() -> str:
    return ConfigOptions().nd2_location


def get_output_file_location() -> str:
    return ConfigOptions().output_location


def get_analysis_location(runlist_fp: str = None) -> str:
    if ConfigOptions().analysis_location is None or ConfigOptions().prior_runlist_fp != runlist_fp:
        tl_dir = get_output_file_location()
        runtime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        if runlist_fp is not None:
            with open(runlist_fp, 'r') as input_runlist:
                runlist_dict = json.load(input_runlist)
                run_name = runlist_dict['run_name']
                replicate_number = runlist_dict['replicate']
                output_path = os.path.join(
                    tl_dir,
                    'analysis_results',
                    f'{run_name}_{replicate_number}_{runtime}',
                )
        else:
            output_path = os.path.join(
                tl_dir,
                'analysis_results',
                f'generic_run_0_{runtime}',
            )
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        ConfigOptions().analysis_location = output_path
        ConfigOptions().prior_runlist_fp = runlist_fp
        return ConfigOptions().analysis_location
    else:
        return ConfigOptions().analysis_location


def beholder_text(input_text: str, color: str = '#49306B'):
    """

    Args:
        input_text:
        color:

    Returns:

    """
    console = Console()
    console.print(input_text, style=color)
