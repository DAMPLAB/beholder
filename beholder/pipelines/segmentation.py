import copy
import csv
import glob
import multiprocessing as mp
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import (
    Tuple,
)

import numpy as np
import tqdm

from beholder.ds import (
    TiffPackage,
)
from beholder.signal_processing import (
    apply_brightness_contrast,
    cellular_highpass_filter,
    combine_frame,
    colorize_frame,
    downsample_image,
    clahe_filter,
    percentile_threshold,
    invert_image,
    erosion_filter,
    remove_background,
    normalize_frame,
    unsharp_mask,
    find_contours,
    generate_arbitrary_stats,
    label_cells,
    draw_mask,
    fluorescence_detection,
    generate_segmentation_visualization,
    debug_image,
    fluorescence_filtration,
)
from beholder.signal_processing.sigpro_utility import (
    get_channel_and_wl_data_from_xml_metadata,
    get_time_stamps_from_xml_metadata,
    ingress_tiff_file,
)
from beholder.utils.config import (
    get_max_processes,
    do_visualization_debug,
    do_render_videos,
    do_single_threaded,
    convert_channel_name_to_color,
)
from beholder.signal_processing.stats import (
    write_stat_record,
    end_of_observation_defocus_clean,
)

import threading
# ----------------------- Command Line Utility Functions -----------------------
def validate_dir_path(input_fp: str):
    if not os.path.isdir(input_fp):
        raise RuntimeError(
            f'Unable to locate {input_fp}. Please check input and try again.'
        )


def generate_mask(input_frame: np.ndarray, contours):
    out_frame = draw_mask(
        input_frame,
        contours,
        colouration='rainbow',
    )
    return out_frame


# ---------------------------- Segmentation Pipeline ---------------------------
def preprocess_primary_frame_and_find_contours(initial_frame: np.ndarray):
    # Each image transform should be giving us back an np.ndarray of the same
    # shape and size.
    # out_frame = signal_transform.lip_removal(initial_frame)
    out_frame = downsample_image(initial_frame)
    out_frame = clahe_filter(out_frame)
    out_frame = percentile_threshold(out_frame)
    out_frame = invert_image(out_frame)
    out_frame = erosion_filter(out_frame)
    out_frame = remove_background(out_frame)
    out_frame = normalize_frame(out_frame)
    out_frame = unsharp_mask(out_frame)
    contours = find_contours(out_frame)
    return contours


def preprocess_color_channel(
        initial_frame: np.ndarray,
        color: str,
        alpha: float = 12,
        beta: int = 0,
):
    out_frame = downsample_image(initial_frame)
    out_frame = apply_brightness_contrast(
        out_frame,
        alpha,
        beta,
    )
    out_frame = colorize_frame(out_frame, color)
    return out_frame


def find_contours_aux_channel(
        aux_frame: np.ndarray,
):
    out_frame = downsample_image(aux_frame)
    out_frame = clahe_filter(out_frame)
    out_frame = percentile_threshold(out_frame)
    out_frame = invert_image(out_frame)
    out_frame = erosion_filter(out_frame)
    out_frame = remove_background(out_frame)
    out_frame = normalize_frame(out_frame)
    out_frame = unsharp_mask(out_frame)
    contours = find_contours(out_frame)
    return contours


def contour_filtration(contours):
    filtered_contours = cellular_highpass_filter(contours)
    return filtered_contours


def generate_frame_visualization(input_arguments: Tuple[int, TiffPackage, str]):
    index, result, filepath = input_arguments
    return generate_segmentation_visualization(
        filename=filepath,
        observation_index=index,
        packed_tiff=result,
    )


def debug_visualization(input_frame: np.ndarray, name: str):
    if do_visualization_debug():
        print(f'Debug for {name}')
        print(f'Total Value for Frames {np.sum(input_frame)}')
        print(f'Shape of Array {input_frame.shape}')
        debug_image(input_frame, name)
        print('------')


def segmentation_pipeline(
        packaged_tiff: TiffPackage,
):
    primary_channel = copy.copy(packaged_tiff.img_array[0])
    auxiliary_channels = copy.copy(packaged_tiff.img_array[1:])
    # ------ SPLITTING OUT THE INPUT DATASTRUCTURE AND INITIAL PROCESSING ------
    for aux_channel_index in range(auxiliary_channels.shape[0]):
        packaged_tiff.cell_signal_auxiliary_frames.append([])
        packaged_tiff.frame_stats.append([])
        packaged_tiff.labeled_auxiliary_frames.append([])
        packaged_tiff.auxiliary_frame_contours.append([])
    for frame_index in range(primary_channel.shape[0]):
        # Handle the primary frame. We use the primary frame to color stuff in.
        primary_frame = primary_channel[frame_index]
        debug_visualization(primary_frame, 'Primary Frame Initial')
        prime_contours = preprocess_primary_frame_and_find_contours(
            primary_frame
        )
        prime_contours = contour_filtration(prime_contours)
        packaged_tiff.primary_frame_contours.append(prime_contours)
        mask_frame = generate_mask(primary_frame, prime_contours)
        packaged_tiff.mask_frames.append(mask_frame)
        # Now we iterate over the other channels.
        aux_frame_container = []
        for aux_channel_index in range(auxiliary_channels.shape[0]):
            # Offset as we assume Channel Zero is our primary (typically grey)
            # frame.
            color = convert_channel_name_to_color(
                packaged_tiff.channel_names[aux_channel_index + 1]
            )
            aux_frame = auxiliary_channels[aux_channel_index][frame_index]
            aux_processed_output = preprocess_color_channel(
                aux_frame,
                color,
            )
            debug_visualization(aux_processed_output, f'Aux Frame {aux_channel_index}')
            aux_frame_container.append(aux_processed_output)
        # -------------------- FIND AUXILIARY FRAME CONTOURS -------------------
        for aux_channel_index in range(auxiliary_channels.shape[0]):
            aux_frame = auxiliary_channels[aux_channel_index][frame_index]
            aux_cont = contour_filtration(find_contours_aux_channel(aux_frame))
            packaged_tiff.auxiliary_frame_contours[aux_channel_index].append(aux_cont)
        # ---------- CORRELATING CELL CONTOURS TO FLUORESCENCE SIGNAL ----------
        for aux_channel_index in range(auxiliary_channels.shape[0]):
            aux_frame = auxiliary_channels[aux_channel_index][frame_index]
            fluorescence_filtration(
                grayscale_frame=primary_frame,
                fluorescent_frame=aux_frame,
                primary_contour_list=prime_contours,
                aux_contour_list=packaged_tiff.auxiliary_frame_contours[aux_channel_index][frame_index]
            )
            correlated_cells = fluorescence_detection(
                primary_frame,
                aux_frame,
                prime_contours,
            )
            packaged_tiff.cell_signal_auxiliary_frames[aux_channel_index].append(correlated_cells)
        frame_stats = generate_arbitrary_stats(
            packaged_tiff.cell_signal_auxiliary_frames
        )
        for aux_channel_index in range(auxiliary_channels.shape[0]):
            packaged_tiff.frame_stats[aux_channel_index].append(frame_stats[aux_channel_index])
        # --------------- LABELING FRAMES WITH DETECTED SIGNALS ----------------
        for aux_channel_index in range(auxiliary_channels.shape[0]):
            aux_frame = auxiliary_channels[aux_channel_index][frame_index]
            debug_visualization(aux_frame, 'Non-Downsampled Aux Frame')
            debug_visualization(downsample_image(aux_frame), 'Downsampled Aux Frame')
            out_label = label_cells(
                downsample_image(aux_frame),
                prime_contours,
                frame_stats[aux_channel_index][frame_index],
            )
            debug_visualization(out_label, 'Out Label')
            packaged_tiff.labeled_auxiliary_frames[aux_channel_index].append(out_label)
        # ---------------- COMBINING FRAMES FOR VISUALIZATION  -----------------
        out_frame = downsample_image(primary_frame)
        for aux_channel_index in range(auxiliary_channels.shape[0]):
            color = convert_channel_name_to_color(
                packaged_tiff.channel_names[aux_channel_index + 1]
            )
            labeled_frame = packaged_tiff.labeled_auxiliary_frames[aux_channel_index][frame_index]
            debug_visualization(labeled_frame, 'Raw Pull of Labeled Frame')
            labeled_frame = colorize_frame(labeled_frame, color)
            debug_visualization(labeled_frame, 'Colorization of Labeled Frame')
            out_frame = combine_frame(
                out_frame,
                labeled_frame,
            )
        debug_visualization(out_frame, 'Final Frame')
        packaged_tiff.final_frames.append(out_frame)
    return packaged_tiff


def enqueue_segmentation(input_fp: str):
    # We should have a top level metadata xml file and then we have a directory
    # called raw_tiffs that has all of the stuff that we really need to work on.
    # We need to take the xml file and extract the channels, sizes, and
    # resolutions and use that to create a class object that can encapuslate the
    # logic related to segmenting tiffs of various dimensions and properties.
    metadata_fp = os.path.join(input_fp, 'metadata.xml')
    tree = ET.parse(metadata_fp)
    # This assumes that everyone has the same amount of channels.
    # If we get to the point where ND2 files have different channels WITHIN
    # themselves I'm throwing my computer into the Charles...
    channels = get_channel_and_wl_data_from_xml_metadata(tree)
    timestamps = get_time_stamps_from_xml_metadata(tree)
    packaged_tiffs = []
    tiff_path = os.path.join(input_fp, 'raw_tiffs')
    tiff_fp = glob.glob(tiff_path + '**/*.tiff')
    sorted_tiffs = sorted(tiff_fp, key=lambda x: int(Path(x).stem))
    # ------------------------------- PACKAGING TIFFS  -------------------------
    for index, tiff_file in tqdm.tqdm(
            enumerate(sorted_tiffs),
            total=len(sorted_tiffs),
            desc="Packaging Tiffs"
    ):
        array = ingress_tiff_file(tiff_file)
        if not array.shape[0]:
            continue
        wavelengths = [x[0] for x in channels[index]]
        channel_names = [x[1] for x in channels[index]]
        ts = [x for x in timestamps[index]]
        title = Path(tiff_file).stem
        inner_pack = TiffPackage(
            img_array=array,
            tiff_name=title,
            channel_names=channel_names,
            channel_wavelengths=wavelengths,
            tiff_index=index,
            timestamps=ts,
        )
        packaged_tiffs.append(inner_pack)
    # ---------------------------- PERFORM SEGMENTATION  -----------------------
    output_location = os.path.join(input_fp, 'segmentation_output')
    if not os.path.exists(output_location):
        os.mkdir(output_location)
    if do_single_threaded():
        segmentation_results = list(
            tqdm.tqdm(
                map(
                    segmentation_pipeline,
                    packaged_tiffs
                ),
                total=len(packaged_tiffs),
                desc="Performing Segmentation"
            )
        )
    else:
        with mp.get_context("spawn").Pool(processes=get_max_processes()) as pool:
            segmentation_results = list(
                tqdm.tqdm(
                    pool.imap(segmentation_pipeline, packaged_tiffs),
                    total=len(packaged_tiffs)
                )
            )
    # ------------------- EXIT GRACEFULLY IF SEGMENTATION FAILED  --------------
    if not segmentation_results:
        return
    # --------------------- PURGE UNWANTED 'DE-FOCUSED' IMAGES  ----------------
    segmentation_results = end_of_observation_defocus_clean(segmentation_results)

    # ---------------- ENSURE OUTPUT DIRECTORY EXISTS AND WRITE OUT  -----------
    # Write out summary statistics
    # Write out more involved channel by channel statistics.
    for i, result in enumerate(tqdm.tqdm(segmentation_results)):
        frame_output = os.path.join(output_location, f'{i + 1}')
        summation_csv_output = os.path.join(frame_output, f'{i + 1}_summary_statistics.csv')
        write_stat_record(input_package=result, record_fp=summation_csv_output)
        for j, channel_result in enumerate(result.cell_signal_auxiliary_frames):
            channel_name = result.channel_names[j + 1]
            csv_location = os.path.join(frame_output, f'{i + 1}_{channel_name}.csv')
            if not os.path.exists(frame_output):
                os.mkdir(frame_output)
            with open(csv_location, 'a') as out_file:
                writer = csv.writer(out_file)
                for frame_result in channel_result:
                    writer.writerow([k.sum_signal for k in frame_result])
    if do_render_videos():
        arg_tuple = range(len(segmentation_results)), segmentation_results, output_location
        if do_single_threaded():
            print(threading.active_count())
            for index, segmentation_result in tqdm.tqdm(
                    enumerate(segmentation_results),
                    leave=False,
                    desc="Generating Frame Results"
            ):
                input_args = index, segmentation_result, output_location
                generate_frame_visualization(input_args)
        else:
            with mp.get_context("spawn").Pool(processes=get_max_processes()) as pool:
                tqdm.tqdm(
                    pool.imap(
                        generate_frame_visualization,
                        arg_tuple,
                    ),
                    desc="Generating Visualizations",
                    total=len(segmentation_results)
                )
