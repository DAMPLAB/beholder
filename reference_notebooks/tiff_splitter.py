'''
--------------------------------------------------------------------------------
Description:

Roadmap:

Written by W.R. Jackson <wrjackso@bu.edu>, DAMP Lab 2020
--------------------------------------------------------------------------------
'''

import click
import tiffile
import tqdm

from signal_processing import (
    signal_transform,
    sigpro_utility,
)


@click.command()
@click.option(
    '--fp',
    default="../data/agarose_pads/SR15_1mM_IPTG_Agarose_TS_1h_1.nd2",
    help='Filepath to Input ND2 files.'
)
@click.option(
    '--context',
    prompt='[multi/single] Would you like individual tiff files for all '
           'channels, or would you like to collapse channels on top of images',
    default='multi',
    help='')
def tiff_splitter(fp: str, context: str):
    # We need a function that takes an ND2, extracts all of the color channels,
    # and returns tuples of each frame with it's constiuent channels as a big
    # ass list
    frames = sigpro_utility.parse_nd2_file(fp)
    channels = sigpro_utility.get_channel_names(fp)
    base_filename = (fp.split("/")[-1])[:-3]
    if (context[0]).lower() == 'm':
        # Iterate over frames, taking each frame and then doing an addweighted
        for i, frame in enumerate(tqdm.tqdm(frames)):
            fn = f'{base_filename}_{i}_multi.tiff'
            base, ch1, ch2 = frame
            ch1 = signal_transform.colorize_frame(ch1, 'green')
            ch2 = signal_transform.colorize_frame(ch2, 'red')
            out_frame = signal_transform.combine_frame(base, ch1)
            out_frame - signal_transform.combine_frame(out_frame, ch2)
            tiffile.imsave(fn, out_frame)
    # Creating a Tiff File for each channel.
    if (context[0]).lower() == 's':
        for i, frame in enumerate(tqdm.tqdm(frames)):
            base_fn = f'{base_filename}_{i}_'
            for j, inner_frame in enumerate(frame):
                fn = f'{base_fn}{channels[j]}.tiff'
                tiffile.imsave(fn, frame)


if __name__ == '__main__':
    tiff_splitter()