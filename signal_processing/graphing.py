'''
--------------------------------------------------------------------------------
Description:

Roadmap:

Written by W.R. Jackson <wrjackso@bu.edu>, DAMP Lab 2020
--------------------------------------------------------------------------------
'''
import random as rng
from typing import (
    List,
)

import numpy as np
import cv2

from signal_processing.stats import CellSignal


def draw_contours(
        input_frame: np.ndarray,
        contour_list: List[np.ndarray],
        stroke: int = -1,
        colouration: str = 'white',
) -> np.ndarray:
    if colouration not in ['white', 'rainbow']:
        print('Current colouration not supported.')
    if colouration == 'white':
        for contour_idx in range(len(contour_list)):
            input_frame = cv2.drawContours(
                input_frame,
                contour_list,
                contour_idx,
                255,
                stroke,
            )
        return input_frame

    if colouration == 'rainbow':
        # TODO: I'm assuming that it's grayscale here. I should make sure to
        #  make this detect our current colorscale.
        rgb_frame = np.zeros(
            (input_frame.shape[0], input_frame.shape[1], 3),
            dtype=np.uint8,
        )
        for contour_idx in range(len(contour_list)):
            color = (
                rng.randint(0, 256),
                rng.randint(0, 256),
                rng.randint(0, 256)
            )
            rgb_frame = cv2.drawContours(
                rgb_frame,
                contour_list,
                contour_idx,
                color,
                stroke,
            )
        return rgb_frame


def write_multiline(
        input_frame: np.ndarray,
        input_str: str,
        loc_x: int,
        loc_y: int,
):
    write_frame = np.zeros(
        (input_frame.shape[0], input_frame.shape[1]),
        dtype=np.uint8,
    )
    cv2.putText(
        write_frame,
        input_str,
        (loc_x, loc_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        255,
        2,
        cv2.LINE_AA,
    )
    output_frame = cv2.addWeighted(
        input_frame,
        1,
        write_frame,
        0.5,
        0,
    )
    return output_frame


def label_cells(
        input_frame: np.ndarray,
        contour_list: List[np.ndarray],
        cell_stats: List[CellSignal],
):
    # Get Bounding Box for Contour
    # Outline Bounding Box in very thin line
    # Next to bounding box put
    # input_frame = np.ndarray(input_frame)
    bbox_list = []
    for contour in contour_list:
        polygon_contour = cv2.approxPolyDP(contour, 3, True)
        bbox_list.append(cv2.boundingRect(polygon_contour))
    for i in range(len(contour_list)):
        # Going to be colorized in a subsequent call.
        c_stats = cell_stats[i]
        input_frame = cv2.rectangle(
            input_frame,
            (bbox_list[i][0], bbox_list[i][1]),
            ((bbox_list[i][0] + bbox_list[i][2]), bbox_list[i][1] + bbox_list[i][3]),
            255,
            1)
        input_frame = write_multiline(
            input_frame,
            f'{c_stats.sum_signal}',
            bbox_list[i][0],
            bbox_list[i][1],
        )
    return input_frame


def generate_frame_report():
    # Calculate Mean Fluorence for channels
    # Calculate Standard Deviation for Channels

    pass
