'''
--------------------------------------------------------------------------------
Description:

Roadmap:

Written by W.R. Jackson <wrjackso@bu.edu>, DAMP Lab 2020
--------------------------------------------------------------------------------
'''
import datetime
import json
import glob
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import (
    List,
)
import numpy as np
from scipy import stats
import pandas as pd
import tqdm

from beholder.utils import (
    BLogger,
)

LOG = BLogger()


def enqueue_lf_analysis(
        input_datasets: List[str],
        calibration_rpu_dataset_fp: str,
        calibration_autofluoresence_dataset_fp: str,
        runlist_fp: str,
):
    """

    Args:
        input_datasets:
        calibration_rpu_dataset_fp:
        calibration_autofluoresence_dataset_fp:
        runlist_fp:

    Returns:

    """
    rpu_df = pd.read_csv(calibration_rpu_dataset_fp)
    af_df = pd.read_csv(calibration_autofluoresence_dataset_fp)
    med_value = rpu_df['fl_median_value'][0]
    max_value = rpu_df['fl_max_value'][0]
    min_value = rpu_df['fl_min_value'][0]
    master_df = pd.DataFrame(
        columns=[
            'datetime',
            'normalized_yfp_fluorescence',
            'normalized_yfp_window_10',
            'max_normalized_yfp_fluorescence',
            'max_normalized_yfp_window_10',
            'group_size',
        ]
    )
    current_end_time = 0
    internal_stats_list = []
    for index, dataset_fp in enumerate(input_datasets):
        segmentation_res_root = os.path.join(dataset_fp, 'segmentation_output')
        result_folders = glob.glob(f'{segmentation_res_root}/*')
        clean_results = []
        dataset_df = pd.DataFrame()
        for res in result_folders:
            if os.path.isdir(res):
                clean_results.append(res)
        clean_results = sorted(clean_results, key=lambda x: int(Path(x).stem))
        for res in tqdm.tqdm(clean_results):
            res_number = Path(res).stem
            stats_file = f'{res_number}_summary_statistics.csv'
            stats_path = os.path.join(res, stats_file)
            internal_stats_df = pd.read_csv(stats_path)
            if internal_stats_df.empty:
                continue
            internal_stats_df['normalized_yfp_fluorescence'] = \
                (internal_stats_df['YFP_fluorescence'] - af_df['fl_median_value']) / med_value
            internal_stats_df['max_normalized_yfp_fluorescence'] = \
                (internal_stats_df['YFP_fluorescence'] - af_df['fl_max_value']) / max_value
            internal_stats_df['min_normalized_yfp_fluorescence'] = \
                (internal_stats_df['YFP_fluorescence'] - af_df['fl_min_value']) / max_value
            internal_stats_df['min_normalized_yfp_fluorescence'] = \
                (internal_stats_df['YFP_fluorescence'] - af_df['fl_min_value']) / max_value
            internal_stats_df['min_normalized_yfp_fluorescence'] = \
                (internal_stats_df['YFP_fluorescence'] - af_df['fl_min_value']) / max_value
            internal_stats_df['timestamps'] = internal_stats_df['timestamps'].astype('int')
            internal_stats_df['source_dataset'] = Path(dataset_fp).stem
            dataset_df = dataset_df.append(
                internal_stats_df[
                    [
                        'timestamps',
                        'normalized_yfp_fluorescence',
                        'max_normalized_yfp_fluorescence',
                        'source_dataset',
                    ]
                ]
            )
        # Below line removes outliers. Should propagate to all other fields of interest.
        dataset_df = dataset_df[(np.abs(stats.zscore(dataset_df['normalized_yfp_fluorescence'])) < 3)]
        # Convert timestamps to something we can do math on.
        dataset_df['datetime'] = pd.to_datetime(dataset_df['timestamps'], unit='s')
        if current_end_time:
            dataset_df['datetime'] = dataset_df['datetime'] + current_end_time
        median_groupby = dataset_df.groupby(
            pd.Grouper(key='datetime', freq='15min')
        )['normalized_yfp_fluorescence']
        maximum_groupby = dataset_df.groupby(
            pd.Grouper(key='datetime', freq='15min')
        )['max_normalized_yfp_fluorescence']
        mean_fl_by_time = median_groupby.agg('mean')
        mean_max_fl_by_time = maximum_groupby.agg('mean')
        group_size = median_groupby.size()
        epoch = datetime.utcfromtimestamp(0)
        sum_stats_df = pd.DataFrame()
        sum_stats_df['datetime'] = (mean_fl_by_time.index.to_pydatetime() - epoch)
        sum_stats_df['normalized_yfp_fluorescence'] = mean_fl_by_time.values
        sum_stats_df['normalized_yfp_window_10'] = sum_stats_df['normalized_yfp_fluorescence'].rolling(10, min_periods=1).median()
        sum_stats_df['max_normalized_yfp_fluorescence'] = mean_max_fl_by_time.values
        sum_stats_df['max_normalized_yfp_window_10'] = sum_stats_df['max_normalized_yfp_fluorescence'].rolling(10, min_periods=1).median()
        sum_stats_df['group_size'] = group_size.values
        # sum_stats_df['source_dataset'] = dataset_df['source_dataset'].values
        current_end_time = sum_stats_df['datetime'].iloc[-1]
        master_df = master_df.append(
            sum_stats_df[
                [
                    'datetime',
                    'normalized_yfp_fluorescence',
                    'normalized_yfp_window_10',
                    'max_normalized_yfp_fluorescence',
                    'max_normalized_yfp_window_10',
                    'group_size',
                ]
            ]
        )
        internal_stats_list.append([Path(dataset_fp).stem, dataset_df])
    tl_dir = Path(input_datasets[0]).parent.absolute()
    runtime = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    with open(runlist_fp, 'r') as input_runlist:
        runlist_dict = json.load(input_runlist)
        if 'run_name' in runlist_dict and "replicate" in runlist_dict:
            run_name = runlist_dict['run_name']
            replicate_number = runlist_dict['replicate']
            output_path = os.path.join(tl_dir, 'analysis_results', f'{run_name}_{replicate_number}_{runtime}')
        else:
            output_path = os.path.join(tl_dir, 'analysis_results', f'{runtime}_results')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    result_path = os.path.join(
        output_path,
        'raw_trend_analysis.csv'
    )
    shutil.copy(runlist_fp, output_path)
    LOG.info(f'Results available at {result_path}')
    master_df.to_csv(result_path)
    for dataset_name, i_df in internal_stats_list:
        out_fp = os.path.join(
            output_path,
            f'{dataset_name}_longform_trend_analysis.csv'
        )
        i_df.to_csv(out_fp)