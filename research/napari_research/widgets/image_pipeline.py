'''
--------------------------------------------------------------------------------
Description:

Roadmap:

Written by W.R. Jackson <wrjackso@bu.edu>, DAMP Lab 2020
--------------------------------------------------------------------------------
'''
import copy
from inspect import signature
from typing import (
    Dict,
    List,
    Tuple,
)
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QComboBox,
    QWidget,
    QSizePolicy,
    QVBoxLayout,
    QGridLayout,
    QLabel,
    QSlider,
    QHBoxLayout,
    QGroupBox,
    QPushButton,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap

from .image_transforms import operations_list

plt.style.use('dark_background')
# Ensure using PyQt5 backend
matplotlib.use('QT5Agg')


class PipelineEditor(QWidget):

    def __init__(self):
        super(QWidget, self).__init__()
        self.layout = QVBoxLayout(self)

        self.label = QLabel('Image Processing Pipeline Editor', self)
        self.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
        self.add_operation = QPushButton('+ Imaging Operation', self)
        self.add_operation.clicked.connect(self.add_new_imaging_operation)
        # self.label.setAlignment(QtCore.Qt.AlignTop)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.add_operation, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom)
        self.dynamic_widgets = {}

    def add_new_imaging_operation(self):
        operation_selection = QComboBox()
        operation_selection.addItems(operations_list.keys())
        operation_selection.currentTextChanged.connect(
            self.on_change,
        )
        operation_selection.setMaximumWidth(200)
        self.generate_grouping(
            [operation_selection],
            'Image Processing Function',
        )

    def generate_grouping(self, widget_list: List, label: str):
        temp_groupbox = QGroupBox(label, checkable=False)
        temp_groupbox.setMaximumWidth(400)
        temp_groupbox.setAlignment(
            QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop
        )
        vbox = QVBoxLayout()
        temp_groupbox.setLayout(vbox)
        for widget in widget_list:
            vbox.addWidget(widget)
        self.layout.addWidget(temp_groupbox)

    def on_change(self, value):
        print(value)
        func = operations_list[value]
        sig = signature(func)
        self.dynamic_widgets = {}
        if sig.return_annotation != np.ndarray:
            print(
                'Non-chainable Operation. Please check return value of input '
                'function.'
            )
        parameter = sig.parameters
        param_keys = sig.parameters.keys()
        for param in param_keys:
            param_type = parameter[param].annotation
            node_str = 'input_node'
            if param[:len(node_str)] == 'input_node':
                self.add_input_node()
            if param_type in [int, float]:
                self.add_numerical_slider()

    def add_numerical_slider(self):
        temp_widget = QSlider(QtCore.Qt.Horizontal, self)
        temp_widget.setMaximum(100)
        temp_widget.setMinimum(0)
        temp_widget.setMaximumWidth(200)
        self.dynamic_widgets[1] = temp_widget
        self.layout.addWidget(temp_widget)

    def add_input_node(self):
        temp_widget = QComboBox()
        temp_widget.addItem('Example Node')
        temp_widget.setEditable(True)
        temp_widget.setMaximumWidth(200)
        line_edit = temp_widget.lineEdit()
        self.dynamic_widgets[0] = temp_widget
        line_edit.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
        self.layout.addWidget(temp_widget)


class MplCanvas(Canvas):
    def __init__(self):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        Canvas.__init__(self, self.fig)
        Canvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        Canvas.updateGeometry(self)
        self.fig.set_facecolor("#00000F")


class ImagePipelineWidget(QWidget):
    def __init__(self):
        super(QWidget, self).__init__()
        self.setGeometry(50, 50, 320, 200)
        self.layout = QGridLayout(self)
        self.canvas = MplCanvas()
        self.editor = PipelineEditor()
        self.layout.addWidget(self.editor, 0, 0, 3, 1)
        self.layout.addWidget(self.canvas, 1, 1, 3, 3)
        self.graph = nx.DiGraph()
        self.loaded_images = {}
        self.initial_nodes = []
        options = {
            'node_color': 'white',
            'node_size': 100,
            'width': 3,
            'with_labels': True,
        }
        self.canvas.fig.set_facecolor('#00000F')
        self.canvas.ax.set_facecolor('#00000F')
        self.graph_positions = {}
        nx.draw(self.graph, ax=self.canvas.ax, **options)

    def perform_pipeline(self, input_args: Dict, transformation_key: str):
        func = operations_list[transformation_key]
        pass

    def generate_initial_network_graph(
            self,
            input_frames: Tuple[np.ndarray, np.ndarray, np.ndarray],
            channel_names: List[str] = None,
    ):
        if channel_names is None:
            channel_names = ['PhlC', 'YfP', 'GfP']
        starting_pos = 0, 0
        y_delta = 10
        for index, frame_and_name in enumerate(zip(input_frames, channel_names)):
            frame, name = frame_and_name
            starting_pos = starting_pos[0], (starting_pos[1] + y_delta)
            self.graph_positions[index] = starting_pos
            self.graph.add_node(index, pos=starting_pos, operation=name, index=index)
            self.initial_nodes.append(index)
            self.loaded_images[index] = frame

    def get_current_head_images(self) -> Dict[str, np.ndarray]:
        # We have starter nodes. We traverse the starter nodes across the graph
        # until we get to pieces with no successors. When that occurs, we know
        # that the image
        master_transform_list = []
        ret_dict = {}
        for node_label in self.initial_nodes:
            internal_transform_list = []
            initial_node = node_label
            current_node_label = node_label
            print(f'{current_node_label=}')
            print(f'{self.graph.adj[current_node_label]=}')
            while list(self.graph.adj[current_node_label].keys()):
                exit_nodes = self.graph.adj[current_node_label].keys()
                print(exit_nodes)
                current_node_label = list(exit_nodes)[0]
                # This assumes that we don't split and always coalesce into a
                # singular image.
                # current_node = list(self.graph.neighbors(node_label))[0]
                internal_transform_list.append(current_node_label)
            master_transform_list.append([initial_node, internal_transform_list])
        print(f'{master_transform_list=}')
        for node_label, transforms in master_transform_list:
            raw_data = self.loaded_images[node_label]
            for transform in transforms:
                print(transform)
                print(operations_list)
                t_func = operations_list[transform]
                raw_data = t_func(raw_data)
            ret_dict[node_label] = raw_data
        return ret_dict



    def update(self):
        print(len(self.graph.nodes))
        if len(self.graph.nodes):
            cdict1 = {
                'blue': ((0.0, 0.0, 1.0),
                         (0.5, 0.1, 0.0),
                         (1.0, 0.0, 0.0)),

                'red': ((0.0, 0.0, 0.0),
                        (0.5, 0.0, 0.1),
                        (1.0, 1.0, 1.0)),

                'green': ((0.0, 0.0, 0.0),
                          (1.0, 0.0, 0.0)),
            }
            cm = LinearSegmentedColormap('BRG', cdict1)
            options = {
                'node_color': range(len(self.graph.nodes)),
                'node_size': 100,
                'width': 3,
                'with_labels': True,
                'cmap': cm,
            }
            nx.draw(
                self.graph,
                nx.get_node_attributes(self.graph, 'pos'),
                ax=self.canvas.ax,
                **options,
            )

    def extend_node_and_edge(
            self,
            prior_node_index: int,
            operation: str,
            x_offset: int = 1,
    ):
        # This is to allow duplicates within a similar list.
        current_node_index = len(self.graph.nodes) + 1
        prior_node = self.graph.nodes.get(prior_node_index)
        x_pos, y_pos = prior_node['pos']
        print(f'{x_pos}')
        print(f'{y_pos}')
        self.graph.add_node(
            current_node_index,
            pos=(x_pos+x_offset, y_pos),
            operation_name=operation,
            index=current_node_index,
        )
        return self.graph.nodes.get(current_node_index)

    def generate_fake_graph(self):
        function_lst = [
            'Downsample Image',
            'Apply Brightness Contrast'
            'Percentile Threshold',
            'Invert Image',
            'Remove Background',
            'Downsample Image',
            'Find Contours',
        ]
        current_node = self.graph.nodes.get(1)
        print(current_node)
        for func in function_lst:
            current_node = self.extend_node_and_edge(
                prior_node_index=current_node['index'],
                operation=func,
            )

