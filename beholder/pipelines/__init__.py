'''
--------------------------------------------------------------------------------
Description:

Roadmap:

Written by W.R. Jackson <wrjackso@bu.edu>, DAMP Lab 2020
--------------------------------------------------------------------------------
'''
from .segmentation import (
    enqueue_segmentation,
)
from .frame_stabilization import (
    enqueue_frame_stabilization,
)
from .nd2_conversion import (
    enqueue_nd2_conversion,
    enqueue_brute_force_conversion,
)

from .panel_detection import (
    enqueue_panel_detection,
)

from .rpu_conversion import (
    enqueue_rpu_calculation,
)

from .longform_trend_analysis import (
    enqueue_lf_analysis,
)

from .generate_figures import (
    enqueue_figure_generation,
)

from .autofluoresence_conversion import (
    enqueue_autofluorescence_calculation,
)

from .gif_generator import (
    enqueue_panel_based_gif_generation,
)

from .wide_analysis import (
    enqueue_wide_analysis,
)

from .long_analysis import (
    enqueue_long_analysis,
)

from .porcelain_check import (
    enqueue_porcelain_conversion,
)

from .split_dataset import (
    enqueue_dataset_split,
)