'''
--------------------------------------------------------------------------------
Description:

Roadmap:

Written by W.R. Jackson <wrjackso@bu.edu>, DAMP Lab 2020
--------------------------------------------------------------------------------
'''
import json
import os
from pathlib import Path
from typing import (
    List,
)

from beholder.pipelines import (
    enqueue_nd2_conversion,
)


def enqueue_dataset_split(
        input_directory: str,
        output_directory: str,
        conversion_list: List[str],
        runlist_fp: str,
):
    """

    Args:
        input_directory:
        output_directory:
        dataset:
        runlist_fp:

    Returns:

    """
    # Parse the runlist and pull out all of the numbers related to this
    # specific dataset.
    # Each child of the dictionary should specify which input dataset is being
    # used to perform the split, and the name of the output.
    # So, something like:
    #     "settings": {
    #       "split_input_dataset": {
    #           "<DATASET_NAME>": {
    #               "<OUTPUT_DATASET_NAME_1>": {
    #                   "panel_numbers": [1, 2, 3, 4, 5...etc]
    #               },
    #               "<OUTPUT_DATASET_NAME_2>": {
    #                   "panel_numbers": [6, 7, 8, 9, 10...etc]
    #               },
    #           }
    #       },
    with open(runlist_fp, 'r') as input_file:
        runlist = json.load(input_file)
        label_dct = runlist['settings']['split_input_dataset']
    for dataset in conversion_list:
        root_name = Path(dataset).stem
        labels = label_dct[root_name]
        for label in labels:
            print(f'{label_dct=}')
            print(f'{label=}')
            panel_numbers = label_dct[root_name][label]['panel_numbers']
            panel_numbers = list(map(int, panel_numbers))
            enqueue_nd2_conversion(
                conversion_list=[dataset],
                output_directory=output_directory,
                force_reconversion=True,
                forced_rename=label,
                split_frames=panel_numbers,
            )
