'''
--------------------------------------------------------------------------------
Description:

Roadmap:

Written by W.R. Jackson <wrjackso@bu.edu>, DAMP Lab 2020
--------------------------------------------------------------------------------
'''
import glob
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import (
    List,
)

import numpy as np
import pandas as pd
import tqdm

from beholder.signal_processing.sigpro_utility import (
    get_channel_data_from_xml_metadata,
    ingress_tiff_file,
)
from beholder.utils import (
    get_analysis_location,
)


def enqueue_long_analysis(
        input_datasets: List[str],
        runlist_fp: str,
):
    # Generate a directory in the analysis results to hold all of the
    # outputted observations.
    # The Name of each of the files will the passed in names of each of the
    # directories.
    # - Tab 1
    # Create a baseline DF, iterating over all of the different panels.
    # Generate a Column for each of the different panels encapsulated within
    # that ND2. Iterate over each of the frames in that TIFF file, and
    # calculate the summation of each of the signals in these.
    output_path = get_analysis_location(runlist_fp)
    input_dest = os.path.join(
        output_path,
        'wide_analysis'
    )
    output_dest = os.path.join(
        output_path,
        'long_analysis',
    )
    channels = ['PhC', 'm-Cherry']
    if not os.path.exists(output_dest):
        os.makedirs(output_dest)
    output_dataframe = pd.DataFrame()
    for channel in channels:
        for nd2_index, dataset_fp in tqdm.tqdm(
                enumerate(input_datasets),
                desc=f'Enumerating over datasets and parsing tiffs...',
                total=len(input_datasets),
        ):
            input_file = os.path.join(input_dest, f'{channel}_{Path(dataset_fp).stem}.xlsx')
            df = pd.read_excel(input_file, sheet_name='Summary Statistics')
            output_dataframe = output_dataframe.append(df, ignore_index=True)
        fp = os.path.join(output_dest, f'{channel}_total_output.xlsx')
        output_dataframe.to_excel(fp)




    # Now we iterate over all of the rows and create a new dude every
    # single time.




