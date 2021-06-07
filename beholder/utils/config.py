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
    nd2_location: str = None
    output_location: str = None

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
                if 'nd2_location' in in_json and "output_location" in in_json:
                    self.nd2_location = in_json['nd2_location']
                    self.output_location = in_json['output_location']


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


def beholder_text(input_text: str, color: str = '#49306B'):
    """

    Args:
        input_text:
        color:

    Returns:

    """
    console = Console()
    console.print(input_text, style=color)
