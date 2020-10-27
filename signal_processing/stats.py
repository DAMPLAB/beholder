'''
--------------------------------------------------------------------------------
Description:

Roadmap:

Written by W.R. Jackson <wrjackso@bu.edu>, DAMP Lab 2020
--------------------------------------------------------------------------------
'''
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

@dataclass
class CellSignal:
    contour_index: int
    sum_signal: float
    fluorescent_pixels: List[float]

def fluoresence_detection(
        grayscale_frame: np.ndarray,
        fluorescent_frame: np.ndarray,
        contour_list: List[np.ndarray],
):
    '''

    Args:
        grayscale_frame:
        fluorescent_frame:
        contour_list:

    Returns:

    '''
    # Cell Locations is a list containing an identifying contour index (so we
    # can map it to cells), the summation of the signal, and then every single
    # fluorescent indice if we feel like doing anything.
    cell_signals = []
    for contour_idx in range(len(contour_list)):
        contour_mask = np.zeros_like(grayscale_frame)
        cv2.drawContours(
            contour_mask,
            contour_list,
            contour_idx,
            color=255,
            thickness=-1
        )
        mask_indices = np.where(contour_mask == 255)
        fluorescent_intensities = fluorescent_frame[mask_indices[0], mask_indices[1]]
        # Noise floor. 256 >> 2 = 32?
        if np.mean(fluorescent_intensities) < 31:
            pass
        else:
            new_cell = CellSignal(
                contour_index=contour_idx,
                sum_signal=np.sum(fluorescent_intensities),
                fluorescent_pixels=fluorescent_intensities,
            )
            cell_signals.append(new_cell)
    return cell_signals


