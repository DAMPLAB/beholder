'''
--------------------------------------------------------------------------------
Description:

Roadmap:

Written by W.R. Jackson <wrjackso@bu.edu>, DAMP Lab 2020
--------------------------------------------------------------------------------
'''
import json
import os
from pathlib import Path

from beholder.pipelines.nd2_conversion import enqueue_nd2_conversion
from beholder.utils import (
    BLogger,
)

log = BLogger()


def enqueue_porcelain_conversion(
        nd2_directory: str,
        output_directory: str,
        batchlist_fp: str = "example_runlist.json",
):
    if not os.path.isfile(batchlist_fp):
        raise RuntimeError('Unable to locate input runlist, please investigate.')
    with open(batchlist_fp, 'r') as input_batch:
        batch_lst = json.load(input_batch)
        runlists = batch_lst['runlists']
        runlist_abs_path = batch_lst['absolute_path']
        runlist_fps = [os.path.join(runlist_abs_path, f'{i}.json') for i in runlists]
        # Checks to see if we expect results from a file we've never converted before.
        for runlist_fp in runlist_fps:
            with open(runlist_fp, 'r') as input_runlist:
                runlist = json.load(input_runlist)
                input_dataset = runlist['input_datasets']
                nd2_file_paths = [os.path.join(nd2_directory, i) + ".nd2" for i in input_dataset]
                file_dirs = [os.path.join(output_directory, i) for i in input_dataset]
                conversion_list = []
                for in_dir, nd2_fp in zip(file_dirs, nd2_file_paths):
                    metadata_path = os.path.join(in_dir, 'metadata.xml')
                    if not os.path.isdir(in_dir):
                        log.info(
                            f'Unable to find directory structure for {Path(in_dir).stem}, '
                            f'attempting conversion to ND2...'
                        )
                        if not os.path.isfile(nd2_fp):
                            log.info(
                                f'Unable to locate nd2 file for {Path(in_dir).stem}, '
                                f'Cowardly exiting...'
                            )
                            exit(1)
                        conversion_list.append(nd2_fp)
                    elif not os.path.isfile(metadata_path):
                        log.info(
                            f'Cannot find metadata file for {Path(in_dir).stem}, '
                            f'performing conversion'
                        )
                        conversion_list.append(nd2_fp)
                    else:
                        continue
                log.warning(f'Missing datasets are {conversion_list}')
                enqueue_nd2_conversion(
                    conversion_list=conversion_list,
                    output_directory=output_directory,
                    force_reconversion=True,
                )
        # Checks to see if a file might've failed in the middle of a conversion,
        # typically when something is started and doesn't finish. We presume
        # that we don't persist the metadata.xml to disk until the completion
        # of the file conversion.
