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


def enqueue_wide_analysis(
        input_datasets: List[str],
        runlist_fp: str,
        calibration_rpu_dataset_fp: str,
        calibration_autofluoresence_dataset_fp: str,
        signal_channel_label: str = 'm-Cherry'
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
    rpu_df = pd.read_csv(calibration_rpu_dataset_fp)
    af_df = pd.read_csv(calibration_autofluoresence_dataset_fp)
    output_path = get_analysis_location(runlist_fp)
    final_dest = os.path.join(
        output_path,
        'wide_analysis'
    )
    if not os.path.exists(final_dest):
        os.makedirs(final_dest)
    for nd2_index, dataset_fp in tqdm.tqdm(
            enumerate(input_datasets),
            desc=f'Enumerating over datasets and parsing tiffs...',
            total=len(input_datasets),
    ):
        tiff_dir_root = os.path.join(dataset_fp, 'raw_tiffs')
        metadata_root = os.path.join(dataset_fp, 'metadata.xml')
        tree = ET.parse(metadata_root)
        channels = get_channel_data_from_xml_metadata(tree)
        raw_sum_df = pd.DataFrame()
        tiff_fps = glob.glob(f'{tiff_dir_root}/*.tiff')
        sorted_tiffs = sorted(tiff_fps, key=lambda x: int(Path(x).stem))
        nd2_label = Path(dataset_fp).stem
        for channel in channels:
            channel_index = channels.index(channel)
            for tiff_fp in tqdm.tqdm(sorted_tiffs):
                tiff = ingress_tiff_file(tiff_fp)
                signal_channel = tiff[channel_index]
                col_label = f'Panel {int(Path(tiff_fp).stem)+1}'
                sum_list = []
                for index, frame in enumerate(signal_channel):
                    sum_list.append(np.sum(signal_channel[index]))
                raw_sum_df[col_label] = sum_list
            summary_stats_df = pd.DataFrame()
            for index, row in raw_sum_df.iterrows():
                row_df = pd.DataFrame()
                # --- RAW ----
                row_df['raw_median_fluorescence'] = [row.median()]
                row_df['raw_mean_fluorescence'] = [row.mean()]
                row_df['raw_max_fluorescence'] = [row.max()]
                row_df['raw_min_fluorescence'] = [row.min()]
                row_df['raw_stddev_fluorescence'] = [row.std()]
                # --- CORRECTED ---
                row_df['corrected_median_fluorescence'] = \
                    ((row.median() - af_df['fl_median_value'])[0]) / rpu_df['fl_median_value']
                row_df['corrected_mean_fluorescence'] = \
                    ((row.mean() - af_df['fl_mean_value'])[0]) / rpu_df['fl_mean_value']
                row_df['corrected_max_fluorescence'] = \
                    ((row.max() - af_df['fl_max_value'])[0]) / rpu_df['fl_max_value']
                row_df['corrected_min_fluorescence'] = \
                    ((row.min() - af_df['fl_min_value'])[0]) / rpu_df['fl_min_value']
                row_df['corrected_stddev_fluorescence'] = \
                    ((row.std() - af_df['fl_std_dev'])[0]) / rpu_df['fl_std_dev']
                summary_stats_df = summary_stats_df.append(row_df, ignore_index=True)
            nd2_dest = os.path.join(final_dest, f'{channel}_{nd2_label}.xlsx')
            with pd.ExcelWriter(nd2_dest) as writer:
                raw_sum_df.to_excel(writer, sheet_name='Raw Panel Stats')
                summary_stats_df.to_excel(writer, sheet_name='Summary Statistics')




    # Now we iterate over all of the rows and create a new dude every
        # single time.




