'''
--------------------------------------------------------------------------------
Description:

Roadmap:

Written by W.R. Jackson <wrjackso@bu.edu>, DAMP Lab 2020
--------------------------------------------------------------------------------
'''
import dataclasses
from typing import (
    List,
    Tuple,
)

import numpy as np
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

    primary_frame_contours: List = None
    auxiliary_frame_contours: List = None

    cell_signal_auxiliary_frames: List = None

    labeled_auxiliary_frames: List = None

    frame_stats: List = None

    stat_file_location: List = None

    def __post_init__(self):
        self.processed_array = []
        self.primary_frame_contours = []
        self.auxiliary_frame_contours = []
        self.cell_signal_auxiliary_frames = []
        self.labeled_auxiliary_frames = []
        self.final_frames = []
        self.mask_frames = []
        self.frame_stats = []
        self.stat_file_location = []

    def get_num_observations(self):
        return self.img_array.shape[1]

    def get_num_frames(
            self,
    ):
        return self.img_array.shape[1]

    def get_num_channels(self):
        """
        TODO: This implicitly believes that everything has been inited,
        so it might be a good idea to have some sort of flag or null check
        installed at a future date.

        Returns:

        """
        return len(self.cell_signal_auxiliary_frames)


@dataclasses.dataclass
class StatisticResults:
    img_array: np.ndarray
    tiff_name: str
    channels: List[str]
    processed_array: List[np.ndarray] = None
