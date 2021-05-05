import multiprocessing as mp
from dataclasses import dataclass
from typing import (
    Dict,

)

from rich.console import Console

from beholder.utils.gof import SingletonBaseClass


@dataclass
class ConfigOptions(metaclass=SingletonBaseClass):
    color_lut: Dict[str, str] = None

    render_videos: bool = True
    max_processes: int = mp.cpu_count() - 1
    single_thread_debug: bool = True
    visualization_debug: bool = False
    test_write: bool = True

    def __post_init__(self):
        if self.color_lut is None:
            self.color_lut = {
                'PhC': 'grey',
                'm-Cherry': 'red',
                'DAPI1': 'green',
                'YFP': 'yellow',
            }


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

def beholder_text(input_text: str, color: str = '#49306B'):
    """

    Args:
        input_text:
        color:

    Returns:

    """
    console = Console()
    console.print(input_text, style=color)