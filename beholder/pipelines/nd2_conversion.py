import glob
import json
import os
from pathlib import Path
from typing import (
    List,
)
from xml.etree import ElementTree as ETree

import numpy as np
import tiffile
import tqdm

from beholder.signal_processing.sigpro_utility import (
    get_channel_and_wl_data_from_xml_metadata,
)
import nd2reader
import pims_nd2


# --------------------------- UTILITY FUNCTIONALITY ----------------------------
def parse_xml_metadata(xml_string, array_order='tyxc'):
    array_order = array_order.upper()
    names, sizes, resolutions = [], [], []
    spatial_array_order = [c for c in array_order if c in 'XYZ']
    size_tags = ['Size' + c for c in array_order]
    res_tags = ['PhysicalSize' + c for c in spatial_array_order]
    metadata_root = ETree.fromstring(xml_string)
    for child in metadata_root:
        if child.tag.endswith('Image'):
            names.append(child.attrib['Name'])
            for grandchild in child:
                if grandchild.tag.endswith('Pixels'):
                    att = grandchild.attrib
                    sizes.append(tuple([int(att[t]) for t in size_tags]))
                    resolutions.append(tuple([float(att[t])
                                              for t in res_tags]))
    return names, sizes, resolutions


def assert_proper_tiff_output_dimensions():
    pass


# ---------------------------- CANONICAL CONVERSION ----------------------------

def enqueue_nd2_conversion(conversion_list: List[str], output_directory: str):
    try:
        import bioformats as bf
        import javabridge
    except ImportError:
        raise RuntimeError(
            'Failed to import bioformats and javabridge. '
            'Please check installation.'
        )
    javabridge.start_vm(class_path=bf.JARS)
    root_logger_name = javabridge.get_static_field(
        "org/slf4j/Logger",
        "ROOT_LOGGER_NAME",
        "Ljava/lang/String;",
    )

    root_logger = javabridge.static_call(
        "org/slf4j/LoggerFactory",
        "getLogger",
        "(Ljava/lang/String;)Lorg/slf4j/Logger;",
        root_logger_name,
    )

    log_level = javabridge.get_static_field(
        "ch/qos/logback/classic/Level",
        "WARN",
        "Lch/qos/logback/classic/Level;",
    )

    javabridge.call(
        root_logger,
        "setLevel",
        "(Lch/qos/logback/classic/Level;)V",
        log_level,
    )
    blank_offset = 0
    for input_fp in conversion_list:
        out_dir = os.path.join(output_directory, Path(input_fp).stem)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        metadata = bf.get_omexml_metadata(input_fp)
        channel_listings = get_channel_and_wl_data_from_xml_metadata(
            ETree.ElementTree(ETree.fromstring(metadata))
        )
        corruption_flag = False
        for channel_listing in channel_listings:
            for channel_entry in channel_listing:
                wavelength, channel_name = channel_entry
                if channel_name == 'DAPI1':
                    print(
                        f'Detected bad channel settings for {input_fp}, '
                        f'falling back to brute-force conversion method... '
                    )
                    corruption_flag = True
                    break
            if corruption_flag:
                break
        if corruption_flag:
            corrupt_metadata_fallback_step_one(
                conversion_fp=input_fp,
                output_directory=output_directory,
                prior_metadata=metadata,
            )
            continue
        image_reader = bf.ImageReader(input_fp, perform_init=True)
        names, sizes, resolutions = parse_xml_metadata(metadata)
        tiff_directory = os.path.join(out_dir, 'raw_tiffs')
        if not os.path.exists(tiff_directory):
            os.mkdir(tiff_directory)
        # We assume uniform shape + size for all of our input frames.
        num_of_frames, x_dim, y_dim, channels = sizes[0]
        for i in tqdm.tqdm(
                range(len(names)),
                desc=f"Converting {Path(input_fp).stem}..."):
            output_array = []
            for j in range(num_of_frames):
                blank_check = image_reader.read(c=1, t=j, series=i)
                if np.sum(blank_check) == 0:
                    blank_offset += 1
                    continue
                else:
                    channel_array = []
                    for k in range(channels):
                        temp = image_reader.read(c=k, t=j, series=i)
                        channel_array.append(temp)
                    output_array.append(channel_array)
            out_array = np.asarray(output_array)
            if len(out_array.shape) == 4:
                out_array = out_array.transpose(1, 0, 2, 3)
            if len(out_array.shape) == 3:
                out_array = out_array.transpose(1, 0, 2)
            save_path = os.path.join(tiff_directory, f'{i}.tiff')
            tiffile.imsave(save_path, out_array)
        metadata_save_path = os.path.join(out_dir, f'metadata.xml')
        # write_xml_metadata(metadata.decode(encoding='utf-8'), metadata_save_path)
        with open(metadata_save_path, 'w') as out_file:
            out_file.write(metadata)
    # print(f'{blank_offset=}')
    javabridge.kill_vm()


# ---------------------------- FALLBACK CONVERSION ----------------------------
def get_channel_name_from_wavelength(wavelength_nm: float):
    wavelength_conversion_lut = {
        '480.0': 'PhC',
        '645.5': 'm-Cherry',
        '535.0': 'GFP',
    }
    return wavelength_conversion_lut[wavelength_nm]


def metadata_correction(prior_metadata: str):
    xml_tree = ETree.ElementTree(ETree.fromstring(prior_metadata))
    metadata_root = xml_tree.getroot()
    for child in metadata_root:
        if child.tag.endswith('Image'):
            for grandchild in child:
                if grandchild.tag.endswith('Pixels'):
                    for greatgrandchild in grandchild:
                        gg_attr = greatgrandchild.attrib
                        if 'EmissionWavelength' not in gg_attr:
                            continue
                        wl = gg_attr['EmissionWavelength']
                        gg_attr['Name'] = get_channel_name_from_wavelength(wl)
    return ETree.tostring(metadata_root, encoding='unicode', method='xml')


def corrupt_metadata_fallback_step_one(
        conversion_fp: str,
        output_directory: str,
        prior_metadata: str,
):
    blank_offset = 0
    base_dir = os.path.join(output_directory, Path(conversion_fp).stem)
    tiff_directory = os.path.join(base_dir, 'raw_tiffs')
    if not os.path.exists(tiff_directory):
        os.mkdir(tiff_directory)
    try:
        with pims_nd2.ND2_Reader(conversion_fp) as input_frames:
            input_frames.iter_axes = 'mt'
            input_frames.bundle_axes = 'xyc'
            # These dimensions are one-indexed, so we correct for this in the
            # loop below.
            observation_count = input_frames.sizes['m']
            time_count = input_frames.sizes['t']
            for i in tqdm.tqdm(
                    range(observation_count),
                    desc=f"Converting {Path(conversion_fp).stem} (Bad Metadata Fallback)..."
            ):
                out_array = []
                for j in range(time_count):
                    blank_check = input_frames[j * observation_count + j][0]
                    if np.sum(blank_check) == 0:
                        blank_offset += 1
                        continue
                    else:
                        out_array.append(input_frames[j * observation_count + j])
                out_array = np.asarray(out_array)
                if len(out_array.shape) == 4:
                    out_array = out_array.transpose(1, 0, 2, 3)
                if len(out_array.shape) == 3:
                    out_array = out_array.transpose(1, 0, 2)
                save_path = os.path.join(tiff_directory, f'{i}.tiff')
                tiffile.imsave(save_path, out_array)
            # We then need to add an editing function that either condenses this down
            # or rewrites the node labels for the xml tree.
            metadata_save_path = os.path.join(base_dir, f'metadata.xml')
            # write_xml_metadata(metadata.decode(encoding='utf-8'), metadata_save_path)
            with open(metadata_save_path, 'w') as out_file:
                corrected_metadata = metadata_correction(prior_metadata)
                out_file.write(corrected_metadata)
    except Exception as e:
        print(
            f'Subsequent metadata corruption detected, manifesting as {e}. '
            f'Falling back to step two...'
        )
        corrupt_metadata_fallback_step_two(
            conversion_fp=conversion_fp,
            output_directory=output_directory,
            prior_metadata=prior_metadata,
        )


def corrupt_metadata_fallback_step_two(
        conversion_fp: str,
        output_directory: str,
        prior_metadata: str,
):
    blank_offset = 0
    base_dir = os.path.join(output_directory, Path(conversion_fp).stem)
    tiff_directory = os.path.join(base_dir, 'raw_tiffs')
    if not os.path.exists(tiff_directory):
        os.mkdir(tiff_directory)
    with nd2reader.ND2Reader(conversion_fp) as input_frames:
        input_frames.iter_axes = 'mt'
        input_frames.bundle_axes = 'xyc'
        # These dimensions are one-indexed, so we correct for this in the
        # loop below.
        observation_count = input_frames.sizes['m']
        time_count = input_frames.sizes['t']
        for i in tqdm.tqdm(
                range(observation_count),
                desc=f"Converting {Path(conversion_fp).stem} (Bad Metadata Fallback)..."
        ):
            out_array = []
            for j in range(time_count):
                blank_check = input_frames[j * observation_count + j][0]
                if np.sum(blank_check) == 0:
                    blank_offset += 1
                    continue
                else:
                    out_array.append(input_frames[j * observation_count + j])
            out_array = np.asarray(out_array)
            if len(out_array.shape) == 4:
                out_array = out_array.transpose(1, 0, 2, 3)
            if len(out_array.shape) == 3:
                out_array = out_array.transpose(1, 0, 2)
            save_path = os.path.join(tiff_directory, f'{i}.tiff')
            tiffile.imsave(save_path, out_array)
        # We then need to add an editing function that either condenses this down
        # or rewrites the node labels for the xml tree.
        metadata_save_path = os.path.join(base_dir, f'metadata.xml')
        # write_xml_metadata(metadata.decode(encoding='utf-8'), metadata_save_path)
        with open(metadata_save_path, 'w') as out_file:
            corrected_metadata = metadata_correction(prior_metadata)
            out_file.write(corrected_metadata)

# --------------------------- BRUTE FORCE CONVERSION ---------------------------
def filter_nonfluorescent_channel(input_frame: np.ndarray):
    if input_frame[0][0][1] == 0.:
        return False
    return True


def enqueue_brute_force_conversion(
        conversion_list: List[str],
        output_directory: str,
        runlist_fp: str,
):
    """

    Args:
        conversion_list:
        output_directory:
        runlist_fp:

    Returns:

    """
    try:
        import bioformats as bf
        import javabridge
    except ImportError:
        raise RuntimeError(
            'Failed to import bioformats and javabridge. '
            'Please check installation.'
        )
    javabridge.start_vm(class_path=bf.JARS)
    root_logger_name = javabridge.get_static_field(
        "org/slf4j/Logger",
        "ROOT_LOGGER_NAME",
        "Ljava/lang/String;",
    )

    root_logger = javabridge.static_call(
        "org/slf4j/LoggerFactory",
        "getLogger",
        "(Ljava/lang/String;)Lorg/slf4j/Logger;",
        root_logger_name,
    )

    log_level = javabridge.get_static_field(
        "ch/qos/logback/classic/Level",
        "WARN",
        "Lch/qos/logback/classic/Level;",
    )

    javabridge.call(
        root_logger,
        "setLevel",
        "(Lch/qos/logback/classic/Level;)V",
        log_level,
    )
    with open(runlist_fp, 'r') as input_file:
        runlist = json.load(input_file)
        observation_counts = runlist['num_observations']
    if len(observation_counts) != len(conversion_list):
        raise RuntimeError(
            'Disconnect between observations and number of datasets.'
        )
    for input_fp, observation_count in zip(conversion_list, observation_counts):
        out_dir = os.path.join(output_directory, Path(input_fp).stem)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        metadata = bf.get_omexml_metadata(input_fp)
        image_reader = bf.ImageReader(input_fp, perform_init=True)
        names, sizes, resolutions = parse_xml_metadata(metadata)
        tiff_directory = os.path.join(out_dir, 'raw_tiffs')
        if os.path.exists(tiff_directory):
            files = glob.glob(os.path.join(tiff_directory, '*.tiff'))
            for f in files:
                os.remove(f)
        if not os.path.exists(tiff_directory):
            os.mkdir(tiff_directory)
        # We assume uniform shape + size for all of our input frames.
        master_array = []
        num_of_frames, x_dim, y_dim, channels = sizes[0]
        for i in tqdm.tqdm(
                range(len(names)),
                desc=f"Generating the Master Array {Path(input_fp).stem}..."):
            for j in range(num_of_frames):
                master_array.append(np.asarray(image_reader.read(t=j, series=i)))
        # We then evenly divide the master array which should have all frames
        # into the number of observations. We're going to have to truncate any
        # of the remainder.
        observation_length = int(len(master_array) / observation_count)
        # We can assume that what we know have is a master array of channels.
        # Some of these are going to be interleaved with dark and light frames,
        # so we have to have some cognizance.
        for i in tqdm.tqdm(
                range(observation_count),
                desc=f'Writing all of the observations to disk.',
        ):
            frame_stride = i * observation_length
            # Sam took a grey shot but not a channel shot and I can't reckon.
            observation_grouping = master_array[
                                   frame_stride:frame_stride+observation_length
                                   ]
            clean_out = list(filter(filter_nonfluorescent_channel, observation_grouping))
            if clean_out:
                clean_out = np.stack(clean_out, axis=0)
                save_path = os.path.join(tiff_directory, f'{i}.tiff')
                clean_out = clean_out.transpose(3, 0, 1, 2)
                tiffile.imsave(save_path, clean_out)
        metadata_save_path = os.path.join(out_dir, f'metadata.xml')
        with open(metadata_save_path, 'w') as out_file:
            corrected_metadata = metadata_correction(metadata)
            out_file.write(corrected_metadata)
    javabridge.kill_vm()

