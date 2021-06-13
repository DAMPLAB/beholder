'''
--------------------------------------------------------------------------------
Description:

Roadmap:

Written by W.R. Jackson <wrjackso@bu.edu>, DAMP Lab 2020
--------------------------------------------------------------------------------
'''
import datetime
import os
from typing import (
    List,
)
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.patches as patches
from beholder.utils import (
    BLogger,
)
from PIL import (
    Image,
    ImageDraw,
)
import imageio

import logging

LOG = BLogger()
logging.getLogger('matplotlib.font_manager').disabled = True


def do_reimport():
    # Some of the Voigt Lab stuff has some global inputs, and this is here
    # to ensure the global traversal of all of the imports doesn't cause any
    # conflicts with out prior settings.
    plt.clf()
    matplotlib.rc('figure', dpi=660)
    # matplotlib.rcParams['pdf.fonttype'] = 42  # for making font editable when exported to PDF for Illustrator
    # matplotlib.rcParams['ps.fonttype'] = 42  # for making font editable when exported to PS for Illustrator
    # plt.rcParams["figure.figsize"] = (40, 20)


dataset_discrimination = ['1_5', '1_9']
dataset_labels = ['SR_1_5', 'SR_1_9']


def generate_lf_analysis_figure(input_fp: str, demarcation_lst: List[str]):
    do_reimport()
    in_file = os.path.join(input_fp, 'longform_trend_analysis.csv')
    if not os.path.exists(in_file):
        raise RuntimeError(f'Cannot find required Longform Analysis csv at location {in_file}')
    df = pd.read_csv(in_file)
    # Assumes that this is a linear amount for observations.
    # df.index = pd.to_datetime(df.index)
    # df['datetime'] = pd.to_datetime(df['timestamps'], unit='s')
    # df.index = df['datetime']
    average = []
    # df['timestamps'] = df['timestamps'] / 900
    rolling_numbers = [5, 10, 50, 100, 200]
    write_list = []
    for rolling_number in rolling_numbers:
        sns.set_theme(style="whitegrid")
        fig = plt.figure(figsize=(22, 5))
        ax1 = fig.add_subplot(111)
        ax1.set_ylim(0.4, 1.2)
        ax1.set_xlim(0, len(df['timestamps']) + 100)
        df.reset_index()
        ax1.set(xlabel='15 Minute Time Intervals', ylabel='YFP Fluorescence (RPUs)')
        plt.title(f'Smoothed Plot for Initial, Window Size {rolling_number}')
        df['YFP_fluorescence'] = df['YFP_fluorescence'].rolling(rolling_number, min_periods=1).median()
        # for index, timestamp in enumerate(df['timestamps'][:-1]):
        #     current_obs = df['timestamps'][index]
        #     next_obs = df['timestamps'][index + 1]
        #     average.append(next_obs - current_obs)
        # print(np.median(average))
        epoch = datetime.datetime.utcfromtimestamp(0)
        df = df[(np.abs(stats.zscore(df['YFP_fluorescence'])) < 3)]
        # df2 = pd.DataFrame(df.groupby([pd.Grouper(key='datetime', freq='60m')]))
        # xs = np.linspace(0, len(df['timestamps']), 1000).reshape(-1, 1)
        # reg = LinearRegression().fit(np.arange(0, len(df['timestamps'])).reshape(-1, 1),
        #                              df['YFP_fluorescence'].to_numpy().reshape(-1, 1))
        # ys = reg.predict(xs)
        ax1 = sns.lineplot(
            x=df['timestamps'],
            y=df['YFP_fluorescence'],
            palette="ch:r=-.2,d=.3_r",
            ax=ax1,
            linewidth=1.5,
        )
        # sns.lineplot(xs.reshape(-1), ys.reshape(-1), color='red', linewidth=2.5, ax=ax1)
        dataset_locations = []
        # for dataset in dataset_discrimination:
        #     dataset_start = float('inf')
        #     start_index = 0
        #     dataset_stop = float('-inf')
        #     stop_index = 0
        #     for index, row in df.iterrows():
        #         # See if substring contained in super string.
        #         if dataset in row['source_dataset']:
        #             if row['timestamps'] < dataset_start:
        #                 dataset_start = row['timestamps']
        #                 start_index = index
        #             if row['timestamps'] > dataset_stop:
        #                 dataset_stop = row['timestamps']
        #                 stop_index = index
        #     dataset_locations.append([start_index, stop_index])
        # for index, loc in enumerate(dataset_locations):
        #     label = dataset_labels[index]
        #     num_spaces = int((loc[1] - loc[0]) / 225)
        #     # ax1.add_patch(
        #     #     patches.Rectangle(
        #     #         xy=(loc[0], 0.425),  # point of origin.
        #     #         width=loc[1] - loc[0],
        #     #         height=10,
        #     #         linewidth=1,
        #     #         color='red',
        #     #         fill=False
        #     #     )
        #     # )
        #     if index == 0:
        #         offset = 2200
        #     if index == 1:
        #         offset = 1800
        #     plt.text(
        #         x=loc[0] + offset,
        #         y=0.425,
        #         s=f'{" "*num_spaces}{label}{" "*num_spaces}',
        #         fontdict=dict(color='white', size=10),
        #         # va='center',
        #         bbox={
        #             'facecolor': 'blue',
        #             'alpha': 0.7,
        #         },
        #         clip_on=False,
        #     )
        #     plt.axvline(x=loc[1], color='k', linestyle='--')
        # np.arange(0, len(df['timestamps']), df['timestamps'].iloc[-1] / 900)
        print(df['timestamps'].iloc[-1])
        plt.xticks(np.arange(0, len(df['timestamps'])+100, 48))
        ax1.xaxis.set_ticklabels([])
        plt.tight_layout()
        # Then we wanted to have some demarcation of events coming through to this
        # point to
        # Let's assume for now, that the points of demarcation will correlate
        # to very indices on the x axis.
        # for demarcation_point in demarcation_lst:
        #     # Math happens here.
        #     plt.axvline(x=demarcation_point, color='k', linestyle='--')
        # plt.draw()
        save_loc = os.path.join(input_fp, f'{rolling_number}_lf_analysis.png')
        plt.savefig(save_loc)
        write_list.append(save_loc)
        plt.cla()
    draw_list = []
    for fp in write_list:
        draw_list.append(Image.open(fp))
    imageio.mimsave(
        os.path.join(input_fp,  'rolling_window_example.gif'),
        draw_list,
    )


def enqueue_figure_generation(input_fp: str, figure_type: str):
    if figure_type == 'longform_analysis':
        LOG.info(f'Generating Longform Analysis figure for {input_fp}')
        generate_lf_analysis_figure(input_fp, demarcation_lst=[])
