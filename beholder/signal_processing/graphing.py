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
from beholder.utils.config import (
    do_single_threaded,
    do_visualization_debug,
)
import threading


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
            'PhC': 'grey',
            'm-Cherry': 'red',
            'DAPI1': 'green',
            'YFP': 'yellow',
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
            keep_composite: bool = False,
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
        if do_visualization_debug():
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
                color=self.color_dict[packed_tiff.channel_names[channel_index+1]],
            )
            plt.plot(
                time_scale,
                median_array,
                color=self.color_dict[packed_tiff.channel_names[channel_index+1]],
                label=f'{packed_tiff.channel_names[channel_index+1]}',
            )
            plt.legend()
        file_handle = f'{self.frame_index}_cell_signal.png'
        fp = os.path.join(self.filepath, file_handle)
        plt.savefig(fp)
        self.cell_signal_image_fp = fp
        plt.clf()
        if do_visualization_debug():
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
        # --------------------------- GREY GRAPHING ----------------------------
        raw_stats: List[CellStats] = packed_tiff.frame_stats[0][self.frame_index]
        raw_cell_count = [len(x.raw_signal) for x in raw_stats]
        raw_cell_count += [0] * (total_num_of_frames - len(raw_cell_count))
        time_scale = list(range(total_num_of_frames))
        time_scale += [0] * (total_num_of_frames - len(raw_cell_count))
        plt.plot(
            time_scale,
            raw_cell_count,
            color=self.color_dict[packed_tiff.channel_names[0]],
            label=f'{packed_tiff.channel_names[0]}',
        )
        # --------------------------- ACTUAL GRAPHING --------------------------
        for channel_index in range(packed_tiff.get_num_channels()):
            cell_stats: List[CellStats] = packed_tiff.frame_stats[channel_index][self.frame_index]
            fl_cell_count = [len(x.fl_signal) for x in cell_stats]
            fl_cell_count += [0] * (total_num_of_frames - len(fl_cell_count))
            time_scale = list(range(total_num_of_frames))
            time_scale += [0] * (total_num_of_frames - len(fl_cell_count))
            # print(f'{cell_count=}, {channel_index=}')
            plt.plot(
                time_scale,
                fl_cell_count,
                color=self.color_dict[packed_tiff.channel_names[channel_index+1]],
                label=f'{packed_tiff.channel_names[channel_index + 1]}',
            )
            plt.legend()
        file_handle = f'{self.frame_index}_cell_count.png'
        fp = os.path.join(self.filepath, file_handle)
        plt.savefig(fp)
        self.cell_count_image_fp = fp
        plt.clf()
        if do_visualization_debug():
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
        fp = os.path.join(self.filepath, file_handle)
        base_image.save(fp)
        self.composite_image_fp = fp
        if do_visualization_debug():
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
    print(f'{threading.active_count()=}')
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
