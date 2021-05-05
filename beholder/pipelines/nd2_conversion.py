import os
from pathlib import Path
from typing import (
    List,
)
from xml.etree import ElementTree as ETree

import numpy as np
import tiffile
import tqdm


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
    for input_fp in conversion_list:
        out_dir = os.path.join(output_directory, Path(input_fp).stem)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        metadata = bf.get_omexml_metadata(input_fp)
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
