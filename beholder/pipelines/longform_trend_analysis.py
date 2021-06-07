'''
--------------------------------------------------------------------------------
Description:

Roadmap:

Written by W.R. Jackson <wrjackso@bu.edu>, DAMP Lab 2020
--------------------------------------------------------------------------------
'''
import os
import glob
from typing import (
    List,
)
from pathlib import Path
import pandas as pd
import tqdm


def enqueue_lf_analysis(
        input_datasets: List[str],
        calibration_rpu_dataset_fp: str,
):
    rpu_df = pd.read_csv(calibration_rpu_dataset_fp)
    med_value = rpu_df['fl_median_value'][0]
    df = pd.DataFrame(columns=['time_start', 'time_stop', 'YFP_fluorescence'])
    for index, dataset_fp in enumerate(input_datasets):
        segmentation_res_root = os.path.join(dataset_fp, 'segmentation_output')
        result_folders = glob.glob(f'{segmentation_res_root}/*')
        clean_results = []
        for res in result_folders:
            if os.path.isdir(res):
                clean_results.append(res)
        clean_results = sorted(clean_results, key=lambda x: int(Path(x).stem))
        time_start = float('inf')
        time_stop = float('-inf')
        # TODO: BAD FORM. DO FIX.
        rolling_sum = 0
        total_res = 0
        for res in tqdm.tqdm(clean_results):
            res_number = Path(res).stem
            stats_file = f'{res_number}_summary_statistics.csv'
            stats_path = os.path.join(res, stats_file)
            df1 = pd.read_csv(stats_path)
            if df1.empty:
                continue
            total_res += 1
            # Perform a correction on overall fluorescence as determined by
            # our median RPU.
            l_time_start = df1['timestamps'].iloc[0]
            l_time_stop = df1['timestamps'].iloc[-1]
            if l_time_start < time_start:
                time_start = l_time_start
            if l_time_stop > time_stop:
                time_stop = l_time_stop
            signal_df = df1.loc[:, df1.columns.str.endswith('_fluorescence')]
            signal_med = signal_df['YFP_fluorescence'].median()
            sig_delta = med_value - signal_med
            df1['YFP_fluorescence'] = df1['YFP_fluorescence'] + sig_delta
            rolling_sum += df1['YFP_fluorescence'].median()
        median_fl = rolling_sum / total_res
        dt = {
            'time_start': [time_start],
            'time_stop': [time_stop],
            'YFP_fluorescence': [median_fl],
        }
        df2 = pd.DataFrame.from_dict(dt)
        df = df.append(df2)
    df.to_csv('test.csv')
