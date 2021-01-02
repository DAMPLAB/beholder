"""
Perform a segmentation and annotate the results with
bounding boxes and text
"""
import numpy as np
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops_table
from skimage.morphology import closing, square, remove_small_objects
import napari
from PyQt5.QtWidgets import (
    QWidget,
    QCheckBox,
    QApplication,
    QTableWidget,
    QHBoxLayout,
    QTableWidgetItem,
    QSizePolicy,
    QVBoxLayout,
    QPushButton,
)
from PyQt5.QtCore import Qt
import sys
from napari.qt import thread_worker
import time
from typing import (
    Tuple,
    Optional,
)
from widgets.image_pipeline import ImagePipelineWidget

COMPUTED_IMAGES = {}
def segment(image):
    """Segment an image using an intensity threshold determined via
    Otsu's method.

    Parameters
    ----------
    image : np.ndarray
        The image to be segmented

    Returns
    -------
    label_image : np.ndarray
        The resulting image where each detected object labeled with a unique integer.
    """
    # apply threshold
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(4))

    # remove artifacts connected to image border
    cleared = remove_small_objects(clear_border(bw), 20)

    # label image regions
    label_image = label(cleared)

    return label_image


def make_bbox(bbox_extents):
    """Get the coordinates of the corners of a
    bounding box from the extents

    Parameters
    ----------
    bbox_extents : list (4xN)
        List of the extents of the bounding boxes for each of the N regions.
        Should be ordered: [min_row, min_column, max_row, max_column]

    Returns
    -------
    bbox_rect : np.ndarray
        The corners of the bounding box. Can be input directly into a
        napari Shapes layer.
    """
    minr = bbox_extents[0]
    minc = bbox_extents[1]
    maxr = bbox_extents[2]
    maxc = bbox_extents[3]

    bbox_rect = np.array(
        [[minr, minc], [maxr, minc], [maxr, maxc], [minr, maxc]]
    )
    bbox_rect = np.moveaxis(bbox_rect, 2, 0)

    return bbox_rect


def circularity(perimeter, area):
    """Calculate the circularity of the region

    Parameters
    ----------
    perimeter : float
        the perimeter of the region
    area : float
        the area of the region

    Returns
    -------
    circularity : float
        The circularity of the region as defined by 4*pi*area / perimeter^2
    """
    circularity = 4 * np.pi * area / (perimeter ** 2)

    return circularity


class TableWidget(QWidget):
    def __init__(self, md):
        super(QWidget, self).__init__()
        self.layout = QHBoxLayout(self)
        self.mdtable = QTableWidget()
        self.layout.addWidget(self.mdtable)

        row_count = len(md)
        col_count = 2
        self.mdtable.setColumnCount(col_count)
        self.mdtable.setRowCount(row_count)

        row = 0

        for key, value in md.items():
            newkey = QTableWidgetItem(key)
            self.mdtable.setItem(row, 0, newkey)
            newvalue = QTableWidgetItem(str(value))
            self.mdtable.setItem(row, 1, newvalue)
            row += 1


# Imports
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
import matplotlib
import matplotlib.pyplot as plt

plt.style.use('dark_background')
# Ensure using PyQt5 backend
matplotlib.use('QT5Agg')


# Matplotlib canvas class to create figure
class MplCanvas(Canvas):
    def __init__(self):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        Canvas.__init__(self, self.fig)
        Canvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        Canvas.updateGeometry(self)


# Matplotlib widget
class MplWidget(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)  # Inherit from QWidget
        self.canvas = MplCanvas()  # Create canvas object
        self.vbl = QVBoxLayout()  # Set box for plotting
        self.vbl.addWidget(self.canvas)
        # self.vbl.addWidget(QCheckBox('Relative Histogram', self))
        # self.vbl.addWidget(QCheckBox('Absolute Histogram', self))
        self.setLayout(self.vbl)
        self.vbl.addWidget(QPushButton('Absolute'))


def parse_slice_data(layer) -> Optional[Tuple[int, int]]:
    slice_t = layer._slice_indices
    if len(slice_t) < 3:
        return None
    else:
        channel, frame_index, *_ = slice_t
        return channel, frame_index


def extract_array_slice(layer, chan_num: int, f_idx: int):
    array = layer.data
    return array[chan_num][f_idx]


def histogram_highest_layer(vwer: napari.Viewer, canvas_ref):
    top_layer = vwer.layers[-1]
    slice_data = parse_slice_data(top_layer)
    if slice_data is None:
        return
    else:
        # print('1')
        # We should add like a cmap to take an arbitrary number of channels in
        # the future, but

        channel, f_index = slice_data
        chan_1 = extract_array_slice(top_layer, 1, f_index)
        chan_2 = extract_array_slice(top_layer, 2, f_index)
        chan_1 = (chan_1 * 65536).round().astype(np.uint16)
        chan_2 = (chan_2 * 65536).round().astype(np.uint16)
        ax = canvas_ref.ax
        ax.clear()
        p_hist, p_bins = np.histogram(chan_1, bins=100)
        r_hist, r_bins = np.histogram(chan_2, bins=100)
        max_bin_size = p_bins[-1] if p_bins[-1] > r_bins[-1] else r_bins[-1]
        center = list(range(0, 100))
        bar_size = 65536 / 100
        ax.bar(p_bins[:len(p_hist)], p_hist, align='edge', width=bar_size, color='green', alpha=0.7)
        ax.bar(r_bins[:len(r_hist)], r_hist, align='edge', width=bar_size, color='red', alpha=0.7)
        ax.set_xlim([0, 20000])
        ax.set_ylim([0, (chan_1.shape[0] * chan_1.shape[1])])
        ax.set_title('Fluorescent Signal')
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Intensity, (a.u. x 10⁴)')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        canvas_ref.draw()



def generate_processing_graph(
        vwer: napari.Viewer,
        pipeline_widget: ImagePipelineWidget,
):
    top_layer = vwer.layers[-1]
    slice_data = parse_slice_data(top_layer)
    if slice_data is None:
        return
    else:
        channel, f_index = slice_data
        print('Hello')
        background = extract_array_slice(top_layer, 0, f_index)
        chan_1 = extract_array_slice(top_layer, 1, f_index)
        chan_2 = extract_array_slice(top_layer, 2, f_index)
        frames = [background, chan_1, chan_2]
        pipeline_widget.generate_initial_network_graph(frames)
        pipeline_widget.update()

def render_processed_frames(
        vwer: napari.Viewer,
        pipeline_widget: ImagePipelineWidget,
):
    top_layer = vwer.layers[-1]
    slice_data = parse_slice_data(top_layer)
    if slice_data is None:
        return None
    else:
        images = pipeline_widget.get_current_head_images()
        pipeline_widget.update()
        return images

viewer = None


def main():
    with napari.gui_qt():
        global viewer
        viewer = napari.Viewer()
        md = {
            'Total Cell Count': '-',
            'Channel 1 Agg. Intensity': '-',
            'Channel 2 Agg. Intensity': '-',
            'Channel 1 Cell Count': '-',
            'Channel 2 Cell Count': '-',
        }
        table_widget = TableWidget(md)
        image_pl_widget = ImagePipelineWidget()
        graph_widget = MplWidget()
        canvas_ref = graph_widget.canvas
        dock_widget = viewer.window.add_dock_widget(
            table_widget,
            name='StatsWindow',
            area='right',
        )
        dock_widget_1 = viewer.window.add_dock_widget(
            graph_widget,
            name='GraphingWidget',
            area='right',
        )
        dock_widget_2 = viewer.window.add_dock_widget(
            image_pl_widget,
            name='PipelineWindow',
            area='bottom',
        )
        data = np.random.random((512, 512))
        layer = viewer.add_image(data)
        active_layer = viewer.layers[0]

        @thread_worker(start_thread=True)
        def layer_update(*, update_period):
            # number of times to update
            nonlocal active_layer
            global viewer
            while True:
                # You need something here to discrimnate between
                # different slices to reduce rerender
                histogram_highest_layer(viewer, canvas_ref)
                if active_layer != viewer.layers[-1]:
                    generate_processing_graph(viewer, image_pl_widget)
                    image_pl_widget.generate_fake_graph()
                    active_layer = viewer.layers[-1]
                    frame_dict = render_processed_frames(viewer, image_pl_widget)
                    if frame_dict is not None:
                        for frame_key in list(frame_dict.keys()):
                            frame_data = frame_dict[frame_key]
                            # viewer.add_layer(frame_key)
                            print(frame_data)
                            print(frame_key)
                            viewer.add_image(frame_data, name=f'{frame_key}')




                # check that data layer is properly assigned and not blocked?
                # while layer.data.all() != dat.all():
                #     layer.data = dat
                yield
        layer_update(update_period=0.10)

if __name__ == '__main__':
    main()