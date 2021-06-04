from pathlib import Path
from typing import (
    List,
)
from xml.etree import ElementTree as ETree

from beholder.signal_processing.sigpro_utility import (
    get_channel_and_wl_data_from_xml_metadata,
)
from beholder.utils import (
    BLogger,
)

log = BLogger()


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

def enqueue_panel_detection(conversion_list: List[str]):
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
    for input_fp in conversion_list:
        log.debug(f'\t | {Path(input_fp).stem}')
    log.debug('--------------')
    for input_fp in conversion_list:
        log.debug(f'Starting Detection for {input_fp}')
        metadata = bf.get_omexml_metadata(input_fp)
        channel_listings = get_channel_and_wl_data_from_xml_metadata(
            ETree.ElementTree(ETree.fromstring(metadata))
        )
        # log.debug(f'Detected Channels for {Path(input_fp).stem}: {channel_listings}')
        names, sizes, resolutions = parse_xml_metadata(metadata)
        log.debug(f'Detected Number of Observations: {len(sizes)}')
        # We assume uniform shape + size for all of our input frames.
        num_of_frames, x_dim, y_dim, channels = sizes[0]
        log.debug(f'Detected Number of Frames: {num_of_frames}')
        log.debug(f'Detected X Dimension: {x_dim}')
        log.debug(f'Detected Y Dimension: {y_dim}')
        log.debug(f'Detected Channels: {channels}')
        log.debug(f'------------------')
    javabridge.kill_vm()

