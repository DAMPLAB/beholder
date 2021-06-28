'''
--------------------------------------------------------------------------------
Description:

Roadmap:

Written by W.R. Jackson <wrjackso@bu.edu>, DAMP Lab 2020
--------------------------------------------------------------------------------
'''
from .gof import SingletonBaseClass
from .config import (
    ConfigOptions,
    do_render_videos,
    do_single_threaded,
    do_visualization_debug,
    get_max_processes,
    beholder_text,
    get_color_keys,
    get_analysis_location,
    convert_channel_name_to_color
)
from .logging import BLogger
