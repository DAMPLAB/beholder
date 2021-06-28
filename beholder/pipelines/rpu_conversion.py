import os

import pandas as pd

from beholder.utils import (
    BLogger,
)

LOG = BLogger()


def enqueue_rpu_calculation(
        rpu_input_fp: str,
        autofluorescence_input_fp: str,
):
    rpu_segmentation_output_top_level = os.path.join(rpu_input_fp, 'segmentation_output')
    af_segmentation_output_top_level = os.path.join(autofluorescence_input_fp, 'autofluorescence_correlation_value.csv')
    top_level_dirs = os.listdir(rpu_segmentation_output_top_level)
    top_level_dirs = list(filter(lambda x: len(x) < 10, top_level_dirs))
    top_level_dirs = sorted(top_level_dirs, key=lambda x: int(x))
    df = pd.DataFrame()
    correction_df = pd.read_csv(af_segmentation_output_top_level)
    for directory in top_level_dirs:
        sum_stat_path = os.path.join(
            rpu_segmentation_output_top_level,
            directory,
            f'{directory}_summary_statistics.csv',
        )
        stat_df = pd.read_csv(sum_stat_path)
        df = pd.concat([df, stat_df])
    # We then calculate the median value of all of the concatened dudes
    out_dict = {
        'fl_median_value': df['YFP_fluorescence'].median() - correction_df['YFP_fluorescence'].median(),
        'fl_mean_value': df['YFP_fluorescence'].mean() - correction_df['YFP_fluorescence'].mean(),
        'fl_min_value': df['YFP_fluorescence'].min() - correction_df['YFP_fluorescence'].min(),
        'fl_max_value': df['YFP_fluorescence'].max() - correction_df['YFP_fluorescence'].max(),
        'std_dev_median_value': df['YFP_std_dev'].median() - correction_df['YFP_std_dev'].median(),
        'std_dev_mean_value': df['YFP_std_dev'].mean() - correction_df['YFP_std_dev'].mean(),
        'cell_count_median_value': df['YFP_cell_count'].median() - correction_df['YFP_cell_count'].median(),
        'cell_count_mean_value': df['YFP_cell_count'].mean() - correction_df['YFP_cell_count'].mean(),
    }
    write_df = pd.DataFrame([out_dict])
    super_summation_path = os.path.join(rpu_input_fp, 'rpu_correlation_value.csv')
    write_df.to_csv(super_summation_path)
    LOG.info(f'Output data available at {super_summation_path}')
