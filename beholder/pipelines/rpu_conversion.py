import os

import pandas as pd


def enqueue_rpu_calculation(input_fp: str):
    segmentation_output_top_level = os.path.join(input_fp, 'segmentation_output')
    top_level_dirs = os.listdir(segmentation_output_top_level)
    top_level_dirs = sorted(top_level_dirs, key=lambda x: int(x))
    df = pd.DataFrame()
    for directory in top_level_dirs:
        print(directory)
        sum_stat_path = os.path.join(segmentation_output_top_level, directory, f'{directory}_summary_statistics.csv')
        print(sum_stat_path)
        stat_df = pd.read_csv(sum_stat_path)
        df = pd.concat([df, stat_df])
    print('Hewwo')
    # We then calculate the median value of all of the concatened dudes
    out_dict = {
        'fl_median_value': df['YFP_fluorescence'].median(),
        'std_dev_median_value': df['YFP_std_dev'].median(),
        'cell_count_median_value': df['YFP_cell_count'].median(),
    }
    write_df = pd.DataFrame([out_dict])
    super_summation_path = os.path.join(input_fp, 'rpu_correlation_value.csv')
    write_df.to_csv(super_summation_path)
