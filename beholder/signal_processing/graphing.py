'''
--------------------------------------------------------------------------------
Description:

Roadmap:

Written by W.R. Jackson <wrjackso@bu.edu>, DAMP Lab 2020
--------------------------------------------------------------------------------
'''
import dataclasses
import os
from pathlib import Path
import random as rng
import imageio
from typing import (
    List,
    Tuple,
    Union,
)
from PIL import (
    Image,
    ImageDraw,
    ImageFont,
)

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_agg import FigureCanvasAgg
import tqdm

from beholder.ds import (
    TiffPackage,
    StatisticResults,
)

from beholder.signal_processing.stats import (
    CellStats,
    CellSignal,
)

SINGLE_THREAD_DEBUG = False


@dataclasses.dataclass
class FrameResult:
    filepath: str
    frame_index: int

    labeled_image_fp: str = None
    cell_signal_image_fp: str = None
    cell_count_image_fp: str = None
    composite_image_fp: str = None

    color_dict: dict = None

    position_dict: dict = None

    def __post_init__(self):
        # TODO: Should probably add a getter or setter somewhere for our color
        #  map so that if we refactor at a future date it'll be consistent.
        self.color_dict = {
            0: 'green',
            1: 'red',
        }
        # This is a bit of a hack to get around some weirdness with multiplots
        # and reducing resolution for some of our combined images as well as
        # giving me some flexibility in terms of future composite imaging when
        # it comes time for publication.
        self.position_dict = {
            'labeled_image_x': 25,
            'labeled_image_y': 50,
            'labeled_image_scale': (600, 600),
            'cell_signal_image_x': 650,
            'cell_signal_image_y': 0,
            'cell_signal_image_scale': (325, 325),
            'cell_count_image_x': 650,
            'cell_count_image_y': 250,
            'cell_count_image_scale': (325, 325),
            'title_position': (50, 10),
        }

    # -------------------------- GETTERS AND SETTERS ---------------------------
    def get_labeled_image(self) -> Image:
        if self.labeled_image_fp is None:
            raise RuntimeError(
                'Labeled Image has not yet been set. Please Investigate'
            )
        return Image.open(self.labeled_image_fp)

    def get_cell_count_image(self) -> Image:
        if self.cell_count_image_fp is None:
            raise RuntimeError(
                'Cell Count Image has not yet been set. Please Investigate'
            )
        return Image.open(self.cell_count_image_fp)

    def get_cell_signal_image(self) -> Image:
        if self.cell_signal_image_fp is None:
            raise RuntimeError(
                'Cell Signal Image has not yet been set. Please Investigate'
            )
        return Image.open(self.cell_signal_image_fp)

    def get_composite_image(self) -> Image:
        if self.composite_image_fp is None:
            raise RuntimeError(
                'Composite has not yet been set. Please Investigate'
            )
        return Image.open(self.composite_image_fp)

    def remove_images(
            self,
            keep_composite: bool = True,
    ):
        written_images = [
            self.labeled_image_fp,
            self.cell_signal_image_fp,
            self.cell_count_image_fp,
        ]
        if not keep_composite:
            written_images.append(self.composite_image_fp)
        if any([image is None for image in written_images]):
            raise RuntimeError('Not all images were visualized')
        [os.remove(image) for image in written_images]

    # ----------------------- VISUALIZATION GENERATION -------------------------

    def load_and_save_labeled_image(
            self,
            packed_tiff: TiffPackage,
    ):
        """

        Args:
            packed_tiff:

        Returns:

        """
        labeled_frame = packed_tiff.final_frames[self.frame_index]
        file_handle = f'{self.frame_index}_labeled_frame.png'
        img = Image.fromarray(labeled_frame, 'RGB')
        fp = os.path.join(self.filepath, file_handle)
        img.save(fp)
        self.labeled_image_fp = fp
        if SINGLE_THREAD_DEBUG:
            self.debug_image(
                self.labeled_image_fp,
                f'Labeled Image Index: {self.frame_index}',
            )

    def generate_cell_signal_graph_and_save(
            self,
            packed_tiff: TiffPackage,
    ):
        total_num_frames = packed_tiff.get_num_frames()
        # ----------------------------- GRAPH SETUP ---------------------------
        plt.clf()
        plt.title('Cellular Signal')
        plt.xlabel('Frame Number')
        plt.ylabel('Intensity A.U.')
        plt.xlim(0, total_num_frames - 1)
        # --------------------------- ACTUAL GRAPHING --------------------------
        for channel_index in range(packed_tiff.get_num_channels()):
            cell_stats: List[CellStats] = packed_tiff.frame_stats[channel_index][self.frame_index]
            median_array = [x.median_signal for x in cell_stats]
            median_stddev = [x.std_dev for x in cell_stats]
            original_length = len(median_array)
            lower_bound = np.subtract(median_array, median_stddev)
            upper_bound = np.add(median_array, median_stddev)
            lower_bound = np.where(lower_bound < 0, 0, lower_bound)
            time_scale = list(range(total_num_frames))
            median_array += [0] * (total_num_frames - original_length)
            lower_bound.resize(total_num_frames, refcheck=False)
            upper_bound.resize(total_num_frames, refcheck=False)
            plt.fill_between(
                time_scale,
                upper_bound,
                lower_bound,
                alpha=.5,
                color=self.color_dict[channel_index],
            )
            plt.plot(
                time_scale,
                median_array,
                color=self.color_dict[channel_index],
            )
        file_handle = f'{self.frame_index}_cell_signal.png'
        fp = os.path.join(self.filepath, file_handle)
        plt.savefig(fp)
        self.cell_signal_image_fp = fp
        plt.clf()
        if SINGLE_THREAD_DEBUG:
            self.debug_image(
                self.cell_signal_image_fp,
                f'Cell Signal Index: {self.frame_index}',
            )

    def generate_cell_count_graph_and_save(
            self,
            packed_tiff: TiffPackage,
    ):
        total_num_of_frames = packed_tiff.get_num_frames()
        # ----------------------------- GRAPH SETUP ---------------------------
        plt.clf()
        plt.title('Cellular Count')
        plt.xlabel('Frame Number')
        plt.ylabel('Num. of Cells')
        plt.xlim(0, total_num_of_frames - 1)
        # --------------------------- ACTUAL GRAPHING --------------------------
        for channel_index in range(packed_tiff.get_num_channels()):
            cell_stats: List[CellStats] = packed_tiff.frame_stats[channel_index][self.frame_index]
            cell_count = [len(x.raw_signal) for x in cell_stats]
            cell_count += [0] * (total_num_of_frames - len(cell_count))
            time_scale = list(range(total_num_of_frames))
            time_scale += [0] * (total_num_of_frames - len(cell_count))
            plt.plot(
                time_scale,
                cell_count,
                color=self.color_dict[channel_index],
            )
        file_handle = f'{self.frame_index}_cell_count.png'
        fp = os.path.join(self.filepath, file_handle)
        plt.savefig(fp)
        plt.savefig(file_handle)
        self.cell_count_image_fp = fp
        if SINGLE_THREAD_DEBUG:
            self.debug_image(
                self.cell_count_image_fp,
                f'Cell Count Index: {self.frame_index}',
            )

    def debug_image(
            self,
            file: Union[str, np.ndarray],
            display_title: str,
    ):
        img = None
        if type(file) == str:
            img = cv2.imread(file, 0)
        if type(file) == np.ndarray:
            img = file
        if img is None:
            raise RuntimeError(
                'Image is never being assigned to a valid input, '
                'please investigate.'
            )
        cv2.imshow(display_title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def draw_composite_image(self):
        # Draw Primary Labeled Microscopy Image
        # 1000 x 500 pixels at 72 DPI = 13.89 x 6.95 Inches
        base_image = Image.new(
            mode='RGBA',
            size=(1000, 500),
            color="white",
        )

        def _draw_on_base(filepath: str, position_name: str):
            img = Image.open(filepath)
            x_pos = self.position_dict[f'{position_name}_x']
            y_pos = self.position_dict[f'{position_name}_y']
            img_scale = self.position_dict[f'{position_name}_scale']
            img.thumbnail(img_scale)
            base_image.paste(img, (x_pos, y_pos))

        _draw_on_base(self.labeled_image_fp, 'labeled_image')
        _draw_on_base(self.cell_count_image_fp, 'cell_count_image')
        _draw_on_base(self.cell_signal_image_fp, 'cell_signal_image')

        text_writer = ImageDraw.Draw(base_image)
        # font = ImageFont.truetype("arial.ttf", 25)
        text_writer.text(
            self.position_dict['title_position'],
            f'{Path(Path(self.filepath).parents[1]).stem}: {self.frame_index}',
            fill=(0, 0, 0),
            # font=font,
        )
        file_handle = f'{self.frame_index}_composite.png'
        base_image.save(file_handle)
        self.composite_image_fp = file_handle
        if SINGLE_THREAD_DEBUG:
            self.debug_image(
                self.composite_image_fp,
                f'Composite Image Index: {self.frame_index}',
            )


@dataclasses.dataclass
class ObservationVisualization:
    observation_index: int
    filename: str
    output_filepath: str = None
    completed_graphs: List[FrameResult] = None

    def __post_init__(self):
        self.completed_graphs = []
        self.output_filepath = os.path.join(
            f'{self.filename}',
            f'{self.observation_index + 1}',
            f'{self.observation_index + 1}_segmentation_visualization.gif'
        )

    def add_frame_result(self, frame_res: FrameResult):
        self.completed_graphs.append(frame_res)

    def generate_animated_video(self):
        draw_list = []
        for graph in self.completed_graphs:
            composite_image = graph.get_composite_image()
            draw_list.append(composite_image)
        imageio.mimsave(
            self.output_filepath,
            draw_list,
        )

    def cleanup_disk(self):
        for graph in self.completed_graphs:
            graph.remove_images()


# ------------------------------------------------------------------------------
def plot_histogram_notebook(input_array: np.ndarray):
    hist, bins = np.histogram(input_array)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    # plt.show()
0

def plot_notebook(input_array: np.ndarray):
    plt.imshow(input_array, cmap='gray')
    # plt.show()


# ------------------------------------------------------------------------------
def draw_contours(
        input_frame: np.ndarray,
        contour_list: List[np.ndarray],
        stroke: int = 3,
) -> np.ndarray:
    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_GRAY2RGB)
    for contour_idx in range(len(contour_list)):
        input_frame = cv2.drawContours(
            input_frame,
            contour_list,
            contour_idx,
            0,
            stroke,
        )
    return input_frame


def draw_mask(
        input_frame: np.ndarray,
        contour_list: List[np.ndarray],
        stroke: int = -1,
        colouration: str = 'white',
) -> np.ndarray:
    if colouration not in ['white', 'rainbow']:
        print('Current colouration not supported.')
    if colouration == 'white':
        for contour_idx in range(len(contour_list)):
            input_frame = cv2.drawContours(
                input_frame,
                contour_list,
                contour_idx,
                255,
                stroke,
            )
        return input_frame

    if colouration == 'rainbow':
        # TODO: I'm assuming that it's grayscale here. I should make sure to
        #  make this detect our current colorscale.
        rgb_frame = np.zeros(
            (input_frame.shape[0], input_frame.shape[1], 3),
            dtype=np.uint8,
        )
        for contour_idx in range(len(contour_list)):
            color = (
                rng.randint(0, 256),
                rng.randint(0, 256),
                rng.randint(0, 256)
            )
            rgb_frame = cv2.drawContours(
                rgb_frame,
                contour_list,
                contour_idx,
                color,
                stroke,
            )
        return rgb_frame


def write_multiline(
        input_frame: np.ndarray,
        input_str: str,
        loc_x: int,
        loc_y: int,
):
    write_frame = np.zeros(
        (input_frame.shape[0], input_frame.shape[1]),
        dtype=np.uint8,
    )
    cv2.putText(
        write_frame,
        input_str,
        (loc_x, loc_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        255,
        2,
        cv2.LINE_AA,
    )
    output_frame = cv2.addWeighted(
        input_frame,
        1,
        write_frame,
        0.5,
        0,
    )
    return output_frame


def label_cells(
        input_frame: np.ndarray,
        contour_list: List[np.ndarray],
        cell_stats: List[CellSignal],
):
    """

    Args:
        input_frame:
        contour_list:
        cell_stats:

    Returns:

    """
    bbox_list = []
    c_signal = sum(cell_stats.raw_signal)
    hist, bins = np.histogram(c_signal, bins=100)
    bin_limit = bins[2]
    for contour in contour_list:
        polygon_contour = cv2.approxPolyDP(contour, 3, True)
        bbox_list.append(cv2.boundingRect(polygon_contour))
    for i in range(len(contour_list)):
        # Going to be colorized in a subsequent call.
        input_frame = cv2.rectangle(
            input_frame,
            (bbox_list[i][0], bbox_list[i][1]),
            ((bbox_list[i][0] + bbox_list[i][2]),
             bbox_list[i][1] + bbox_list[i][3]),
            255,
            1)
        input_frame = write_multiline(
            input_frame,
            f'{cell_stats.raw_signal[i]}',
            bbox_list[i][0],
            bbox_list[i][1],
        )
    return input_frame


def plot_total(
        total_statistics: List[Tuple[float, float, float]],
        context: str = 'save',
):
    '''

    Args:
        total_statistics:

    Returns:

    '''
    channel_one_stats = []
    channel_two_stats = []
    out_list = []
    fig = plt.figure()
    canvas = FigureCanvasAgg(fig)
    for stat_pair in total_statistics:
        c1_stat, c2_stat = stat_pair
        channel_one_stats.append(c1_stat)
        channel_two_stats.append(c2_stat)
    channel_one_median = [stat[0] for stat in channel_one_stats]
    channel_one_std_dev = [stat[1] for stat in channel_one_stats]
    channel_one_cell_count = [stat[2] for stat in channel_one_stats]
    channel_two_median = [stat[0] for stat in channel_two_stats]
    channel_two_std_dev = [stat[1] for stat in channel_two_stats]
    channel_two_cell_count = [stat[2] for stat in channel_two_stats]
    channel_one_pos = np.add(channel_one_median, channel_one_std_dev)
    channel_one_neg = np.subtract(channel_one_median, channel_one_std_dev)
    channel_two_pos = np.add(channel_two_median, channel_two_std_dev)
    channel_two_neg = np.subtract(channel_two_median, channel_two_std_dev)
    time_scale = range(len(channel_one_stats))
    # We want the lower band, the higher band, and the actual value.
    ax1.fill_between(
        time_scale,
        channel_one_pos,
        channel_one_neg,
        alpha=.5,
        color='green',
    )
    ax1.fill_between(
        time_scale,
        channel_two_pos,
        channel_two_neg,
        alpha=.5,
        color='red',
    )
    ax1.plot(time_scale[::-1], channel_one_median[::-1], color='green')  #
    ax1.plot(time_scale[::-1], channel_two_median[::-1], color='red')  #
    if context == 'save':
        plt.savefig('example.png')
    if context == 'subplot':
        canvas.draw()
        buf = canvas.buffer_rgba()
        buf_1 = buf[:]
        out_list.append(buf_1)
    # plt.clf()
    # plt.cla()
    # plt.close()
    ax2.plot(time_scale[::-1], channel_one_cell_count[::-1], color='blue')  #
    ax2.plot(time_scale[::-1], channel_two_cell_count[::-1], color='purple')  #
    if context == 'save':
        plt.savefig('example1.png')
    if context == 'subplot':
        canvas.draw()
        buf = canvas.buffer_rgba()
        buf_2 = buf[:]
        out_list.append(buf_2)
    # plt.legend()
    # plt.show()
    if context == 'subplot':
        canvas.draw()
        buf = canvas.buffer_rgba()
        return buf


def generate_multiplot(
        frame_list: List[np.ndarray],
        frame_annotations: List[str],
        context: str = 'active_frame',
):
    fig = plt.figure(figsize=(16, 3))
    canvas = FigureCanvasAgg(fig)
    gs1 = gridspec.GridSpec(1, 4)
    gs1.update(wspace=0.2, hspace=0.02)
    plt.margins(.5, .5)
    for i, anno in enumerate(frame_annotations):
        ax1 = plt.subplot(gs1[i])
        active_frame = frame_list[i]
        if not i:
            ax1.imshow(active_frame[
                       int(active_frame.shape[0] / 2):int(active_frame.shape[0]),
                       int(active_frame.shape[1] / 2):int(active_frame.shape[1])
                       ],
                       cmap='gray')
        else:
            ax1.imshow(active_frame[
                       int(active_frame.shape[0] / 2):int(active_frame.shape[0]),
                       int(active_frame.shape[1] / 2):int(active_frame.shape[1])
                       ],
                       interpolation='nearest')
        ax1.set_title(anno)
        plt.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
        autoAxis = ax1.axis()
        rec = plt.Rectangle((autoAxis[0] - 0.7, autoAxis[2] - 0.2),
                            (autoAxis[1] - autoAxis[0]) + 1,
                            (autoAxis[3] - autoAxis[2]) + 0.4, fill=False, lw=2)
        rec = ax1.add_patch(rec)
        rec.set_clip_on(False)
    plt.axis('off')
    if context == 'to_disk':
        plt.savefig(
            "test1.jpg",
            bbox_inches='tight',
            pad_inches=.2,
        )
    if context == 'active_frame':
        canvas.draw()
        buf = canvas.buffer_rgba()
        return np.asarray(buf)


def stat_splitter(
        total_statistics: List[Tuple[float, float]],
):
    channel_one_stats = []
    channel_two_stats = []
    for stat_pair in total_statistics:
        c1_stat, c2_stat = stat_pair
        channel_one_stats.append(c1_stat)
        channel_two_stats.append(c2_stat)
    return channel_one_stats, channel_two_stats


# It's a List of CellSignals, avoiding a circular import.
def plot_cell_signal(
        cell_signal_result: List[CellSignal],
        channel_index: int,
        input_axis,
        observation_length: int
):
    """

    Args:
        cell_signal_results:

    Returns:

    """
    # TODO: I hate this. Bad data gonna bad data.
    # For future Jackson, Sam's test data has a bunch of mislabled channels
    # colors. For example, a number of them claim to be DAPI when they are in
    # fact mCherry or similar. So, we would assume that the correct behavior
    # would be to colorize the frame to correspond to it's fluorescent channel,
    # to emulate what someone would see on a microscopic instrument. However,
    # if we do that we're going to get nothing but bitchy complaints from
    # everyone about how they didn't actually mess with the microscope and the
    # data is wrong and etc etc and I just don't really have the patience.
    color_dict = {
        0: 'green',
        1: 'red',
    }
    color = color_dict[channel_index]
    # canvas = FigureCanvasAgg(input_axis)
    input_axis.set_title('Cellular Signal')
    input_axis.set_xlabel('Observation Number')
    input_axis.set_ylabel('Intensity A.U.')
    input_axis.set_xlim(0, observation_length)
    median_array = [x.median_signal for x in cell_signal_result]
    original_delta = len(median_array)
    median_stddev = [x.std_dev for x in cell_signal_result]
    lower_bound = np.subtract(median_array, median_stddev)
    upper_bound = np.add(median_array, median_stddev)
    lower_bound = np.where(lower_bound < 0, 0, lower_bound)
    time_scale = list(range(observation_length))
    median_array += [0] * (observation_length - original_delta)
    lower_bound.resize(observation_length, refcheck=False)
    upper_bound.resize(observation_length, refcheck=False)
    # We want the lower band, the higher band, and the actual value.
    input_axis.fill_between(
        time_scale,
        upper_bound,
        lower_bound,
        alpha=.5,
        color=color,
    )
    input_axis.plot(time_scale, median_array, color=color)
    plt.draw()


def plot_cell_count(
        cell_signal_result: List[CellSignal],
        channel_index: int,
        input_axis,
        observation_length: int
):
    input_axis.set_title('Cell Count')
    cell_count = [len(x.raw_signal) for x in cell_signal_result]
    cell_count += [0] * (observation_length - len(cell_count))
    time_scale = list(range(observation_length))
    time_scale += [0] * (observation_length - len(cell_count))
    color_dict = {
        0: 'green',
        1: 'red',
    }
    color = color_dict[channel_index]
    # We want the lower band, the higher band, and the actual value.
    input_axis.plot(time_scale, cell_count, color=color)


def plot_cellular_signal(channel_one_stats, channel_two_stats):
    fig = plt.figure()
    canvas = FigureCanvasAgg(fig)
    plt.title('Cellular Signal')
    channel_one_median = [stat[0] for stat in channel_one_stats]
    channel_one_std_dev = [stat[1] for stat in channel_one_stats]
    channel_two_median = [stat[0] for stat in channel_two_stats]
    channel_two_std_dev = [stat[1] for stat in channel_two_stats]
    channel_one_pos = np.add(channel_one_median, channel_one_std_dev)
    channel_one_neg = np.subtract(channel_one_median, channel_one_std_dev)
    channel_two_pos = np.add(channel_two_median, channel_two_std_dev)
    channel_two_neg = np.subtract(channel_two_median, channel_two_std_dev)
    channel_one_neg = np.where(channel_one_neg < 0, 0, channel_one_neg)
    channel_two_neg = np.where(channel_two_neg < 0, 0, channel_two_neg)
    time_scale = range(len(channel_one_stats))
    # We want the lower band, the higher band, and the actual value.
    plt.fill_between(
        time_scale,
        channel_one_pos,
        channel_one_neg,
        alpha=.5,
        color='green',
    )
    plt.fill_between(
        time_scale,
        channel_two_pos,
        channel_two_neg,
        alpha=.5,
        color='red',
    )
    plt.plot(time_scale, channel_one_median, color='green')  #
    plt.plot(time_scale, channel_two_median, color='red')  #
    canvas.draw()
    buf = canvas.buffer_rgba()
    return buf


# def plot_signal_histogram(
#         segmentation_result,
#         observation_index,
#         frame_index,
#         axis,
#         plot_name,
# ):
#     '''
#
#     Returns:
#
#     '''
#     axis.set_title(plot_name)
#     for channel_index, channel in enumerate(segmentation_result.cell_signal_auxiliary_frames):
#         print(frame_index)
#         p_hist, p_bins = np.histogram(segmentation_result.cell_signal_auxiliary_frames[observation_index][
#         channel_index][frame_index], bins=100)
#         center = (p_bins[:-1] + p_bins[1:]) / 2
#         color_dict = {
#             0: 'green',
#             1: 'red',
#         }
#         color = color_dict[channel_index]
#         axis.bar(center, p_hist, align='center', width=100, color=color)
#     axis.set_xlim([0, 100])
#     axis.set_ylim([0, 1000000])
#     axis.set_xticklabels([])
#     axis.set_yticklabels([])


def generate_image_canvas(
        processed_frame: np.ndarray,
        raw_frame: np.ndarray,
        stats_list: List,
        title: str,
):
    fig = plt.figure(figsize=(13, 6))
    canvas = FigureCanvasAgg(fig)
    grid = plt.GridSpec(7, 6, hspace=0.0, wspace=0.0)
    plt.title(f'{title}')
    # Padding the Array to our final point
    # stats_list += [[(0, 0, 0), (0, 0, 0)]] * (stats_final_size - len(stats_list))
    axis_1 = fig.add_subplot(grid[:6, :5])
    axis_2 = fig.add_subplot(grid[0:3, 4:6])
    axis_3 = fig.add_subplot(grid[3:6, 4:6])
    axis_4 = fig.add_subplot(grid[6:7, :])
    # axis_5 = fig.add_subplot(grid[6:7, 3:6])
    c1_stats, c2_stats = stat_splitter(stats_list)
    cell_signal = plot_cellular_signal(c1_stats, c2_stats)
    cell_count = plot_cell_count(c1_stats, c2_stats)
    # TODO: Work via Mutation.
    plot_signal_histogram(processed_frame, raw_frame, axis_4, 'Processed Frame')
    plot_histogram_notebook(raw_frame)
    array_list = [processed_frame, cell_signal, cell_count]
    axis_list = [axis_1, axis_2, axis_3]
    for array, axis in zip(array_list, axis_list):
        axis.imshow(array, interpolation='nearest')
        plt.axis('off')
        axis.set_xticklabels([])
        axis.set_yticklabels([])
        axis.set_aspect('equal')
    plt.tight_layout()
    plt.text(0.1, 0.9, 'matplotlib', ha='center', va='center')
    canvas.draw()
    buf = canvas.buffer_rgba()
    plt.close(fig)
    return np.asarray(buf)


def generate_segmentation_visualization(
        filename: str,
        observation_index: int,
        packed_tiff: TiffPackage,
):
    """

    Args:
        filename:
        observation_index:
        packed_tiff:

    Returns:

    """
    out_list = []
    obs_viz = ObservationVisualization(
        observation_index=observation_index,
        filename=filename,
    )
    for frame_index in tqdm.tqdm(range(packed_tiff.get_num_frames())):
        frame_res = FrameResult(
            filepath=os.path.join(filename, f'{observation_index + 1}'),
            frame_index=frame_index,
        )
        frame_res.load_and_save_labeled_image(
            packed_tiff=packed_tiff,
        )
        frame_res.generate_cell_signal_graph_and_save(
            packed_tiff=packed_tiff,
        )
        frame_res.generate_cell_count_graph_and_save(
            packed_tiff=packed_tiff,
        )
        frame_res.draw_composite_image()
        obs_viz.add_frame_result(frame_res)
        out_list.append(obs_viz.output_filepath)
    obs_viz.generate_animated_video()
    obs_viz.cleanup_disk()
