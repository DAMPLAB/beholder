import os

import pandas as pd

from beholder.utils import (
    BLogger,
)

LOG = BLogger()


def enqueue_autofluorescence_calculation(input_fp: str):
    segmentation_output_top_level = os.path.join(input_fp, 'segmentation_output')
    top_level_dirs = os.listdir(segmentation_output_top_level)
    top_level_dirs = list(filter(lambda x: len(x) < 10, top_level_dirs))
    top_level_dirs = sorted(top_level_dirs, key=lambda x: int(x))
    df = pd.DataFrame()
    for directory in top_level_dirs:
        sum_stat_path = os.path.join(
            segmentation_output_top_level,
            directory,
            f'{directory}_summary_statistics.csv',
        )
        stat_df = pd.read_csv(sum_stat_path)
        df = pd.concat([df, stat_df])
    # We then calculate the median value of all of the concatened dudes
    out_dict = {
        'fl_median_value': df['YFP_fluorescence'].median(),
        'fl_mean_value': df['YFP_fluorescence'].mean(),
        'fl_min_value': df['YFP_fluorescence'].min(),
        'fl_max_value': df['YFP_fluorescence'].max(),
        'std_dev_median_value': df['YFP_std_dev'].median(),
        'std_dev_mean_value': df['YFP_std_dev'].mean(),
    }
    write_df = pd.DataFrame([out_dict])
    super_summation_path = os.path.join(input_fp, 'autofluorescence_correlation_value.csv')
    write_df.to_csv(super_summation_path)
    LOG.info(f'Output data available at {super_summation_path}')
