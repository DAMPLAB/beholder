'''
--------------------------------------------------------------------------------
Description:

Roadmap:

Written by W.R. Jackson <wrjackso@bu.edu>, DAMP Lab 2020
--------------------------------------------------------------------------------
'''
import os
from typing import (
    List,
)
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from beholder.utils import (
    BLogger,
)

LOG = BLogger()


def do_reimport():
    # Some of the Voigt Lab stuff has some global inputs, and this is here
    # to ensure the global traversal of all of the imports doesn't cause any
    # conflicts with out prior settings.
    plt.clf()
    matplotlib.rc('figure', dpi=160)
    matplotlib.rcParams['pdf.fonttype'] = 42  # for making font editable when exported to PDF for Illustrator
    matplotlib.rcParams['ps.fonttype'] = 42  # for making font editable when exported to PS for Illustrator


def generate_lf_analysis_figure(input_fp: str, demarcation_lst: List[str]):
    do_reimport()
    in_file = os.path.join(input_fp, 'longform_trend_analysis.csv')
    if not os.path.exists(in_file):
        raise RuntimeError(f'Cannot find required Longform Analysis csv at location {in_file}')
    df = pd.read_csv(in_file)
    # Assumes that this is a linear amount for observations.
    median_time_list = df['time_stop'] - df['time_start']
    plt.scatter(
        x=median_time_list,
        y=df['YFP_fluorescence'],
        s=200,
        edgecolors='k',
        linewidths=2,
        color='blue',
    )
    line_of_best_fit = np.logspace(
        np.log(df['YFP_fluorescence'].iloc[0]),
        np.log(df['YFP_fluorescence'].iloc[-1]),
        1000,
    )
    # Then we wanted to have some demarcation of events coming through to this
    # point to
    # Let's assume for now, that the points of demarcation will correlate
    # to very indices on the x axis.
    for demarcation_point in demarcation_lst:
        # Math happens here.
        plt.axvline(x=demarcation_point, color='k', linestyle='--')
    plt.show()
    save_loc = os.path.join(input_fp, 'lf_analysis.png')
    plt.savefig(save_loc)


def enqueue_figure_generation(input_fp: str, figure_type: str):
    if figure_type == 'longform_analysis':
        LOG.info(f'Generating Longform Analysis figure for {input_fp}')
        generate_lf_analysis_figure(input_fp, demarcation_lst=[])
