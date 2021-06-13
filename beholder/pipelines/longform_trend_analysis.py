'''
--------------------------------------------------------------------------------
Description:

Roadmap:

Written by W.R. Jackson <wrjackso@bu.edu>, DAMP Lab 2020
--------------------------------------------------------------------------------
'''
import datetime
import glob
import os
import shutil
import time
from pathlib import Path
from typing import (
    List,
)

import pandas as pd
import tqdm

from beholder.utils import (
    BLogger,
)
from datetime import datetime

LOG = BLogger()
from sklearn.preprocessing import scale


def enqueue_lf_analysis(
        input_datasets: List[str],
        calibration_rpu_dataset_fp: str,
        runlist_fp: str,
):
    rpu_df = pd.read_csv(calibration_rpu_dataset_fp)
    med_value = rpu_df['fl_median_value'][0]
    df = pd.DataFrame(columns=['timestamps', 'YFP_fluorescence', 'source_dataset'])
    for index, dataset_fp in enumerate(input_datasets):
        segmentation_res_root = os.path.join(dataset_fp, 'segmentation_output')
        result_folders = glob.glob(f'{segmentation_res_root}/*')
        clean_results = []
        for res in result_folders:
            if os.path.isdir(res):
                clean_results.append(res)
        clean_results = sorted(clean_results, key=lambda x: int(Path(x).stem))
        # TODO: BAD FORM. DO FIX.
        for res in tqdm.tqdm(clean_results):
            res_number = Path(res).stem
            stats_file = f'{res_number}_summary_statistics.csv'
            stats_path = os.path.join(res, stats_file)
            df1 = pd.read_csv(stats_path)
            if df1.empty:
                continue
            df1['YFP_fluorescence'] = df1['YFP_fluorescence'] / med_value
            df1['timestamps'] = df1['timestamps'].astype('int')
            df1['source_dataset'] = Path(dataset_fp).stem
            df = df.append(df1[['YFP_fluorescence', 'timestamps', 'source_dataset']])
    df['datetime'] = pd.to_datetime(df['timestamps'], unit='s')
    gb = df.groupby(pd.Grouper(key='datetime', freq='15min'))['YFP_fluorescence']
    mean_fl_by_time = gb.agg('mean')
    group_size = gb.size()
    epoch = datetime.utcfromtimestamp(0)
    df3 = pd.DataFrame()
    df3['datetime'] = (mean_fl_by_time.index.to_pydatetime() - epoch)
    df3['YFP_fluorescence'] = mean_fl_by_time.values
    df3['group_size'] = group_size.values
    tl_dir = Path(input_datasets[0]).parent.absolute()
    runtime = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    output_path = os.path.join(tl_dir, 'analysis_results', f'{runtime}_results')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    result_path = os.path.join(
        output_path,
        'old_longform_trend_analysis.csv'
    )
    new_result_path = os.path.join(
        output_path,
        'longform_trend_analysis.csv'
    )
    shutil.copy(runlist_fp, output_path)
    LOG.info(f'Results available at {result_path}')
    df.to_csv(result_path)
    df3.to_csv(new_result_path)



# def enqueue_lf_analysis(
#         input_datasets: List[str],
#         calibration_rpu_dataset_fp: str,
#         runlist_fp: str,
# ):
#     rpu_df = pd.read_csv(calibration_rpu_dataset_fp)
#     med_value = rpu_df['fl_median_value'][0]
#     df = pd.DataFrame(columns=['time_start', 'time_stop', 'YFP_fluorescence'])
#     offset_datetime = 0
#     for index, dataset_fp in enumerate(input_datasets):
#         segmentation_res_root = os.path.join(dataset_fp, 'segmentation_output')
#         result_folders = glob.glob(f'{segmentation_res_root}/*')
#         clean_results = []
#         for res in result_folders:
#             if os.path.isdir(res):
#                 clean_results.append(res)
#         clean_results = sorted(clean_results, key=lambda x: int(Path(x).stem))
#         time_start = float('inf')
#         time_stop = float('-inf')
#         # TODO: BAD FORM. DO FIX.
#         rolling_sum = 0
#         total_res = 0
#         for res in tqdm.tqdm(clean_results):
#             res_number = Path(res).stem
#             stats_file = f'{res_number}_summary_statistics.csv'
#             stats_path = os.path.join(res, stats_file)
#             df1 = pd.read_csv(stats_path)
#             if df1.empty:
#                 continue
#             total_res += 1
#             # Perform a correction on overall fluorescence as determined by
#             # our median RPU.
#             l_time_start = df1['timestamps'].iloc[0]
#             l_time_stop = df1['timestamps'].iloc[-1]
#             if l_time_start < time_start:
#                 time_start = l_time_start
#             if l_time_stop > time_stop:
#                 time_stop = l_time_stop
#             # THE CORRECTION
#             df1['YFP_fluorescence'] = df1['YFP_fluorescence'] / med_value
#             rolling_sum += df1['YFP_fluorescence'].mean()
#             total_res += 1
#         if not total_res:
#             continue
#         median_fl = rolling_sum / total_res
#         dt = {
#             'time_start': [time_start + offset_datetime],
#             'time_stop': [time_stop + offset_datetime],
#             'YFP_fluorescence': [median_fl],
#         }
#         df2 = pd.DataFrame.from_dict(dt)
#         df = df.append(df2)
#         offset_datetime = time_stop
#     tl_dir = Path(input_datasets[0]).parent.absolute()
#     print(1)
#     runtime = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
#     output_path = os.path.join(tl_dir, 'analysis_results', f'{runtime}_results')
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)
#     print(2)
#     result_path = os.path.join(
#         output_path,
#         'longform_trend_analysis.csv'
#     )
#     shutil.copy(runlist_fp, output_path)
#     LOG.info(f'Results available at {result_path}')
#     df.to_csv(result_path)
