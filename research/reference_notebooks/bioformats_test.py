import bioformats as bf
import javabridge
import bioformats
from xml.etree import ElementTree as ETree
import numpy as np
from matplotlib import pyplot as plt, cm
import tiffile
import os
javabridge.start_vm(class_path=bioformats.JARS)
DEFAULT_DIM_ORDER = 'tyxc'
from pathlib import Path
import glob

def parse_xml_metadata(xml_string, array_order=DEFAULT_DIM_ORDER):
    """Get interesting metadata from the LIF file XML string.
    Parameters
    ----------
    xml_string : string
        The string containing the XML data.
    array_order : string
        The order of the dimensions in the multidimensional array.
        Valid orders are a permutation of "tzyxc" for time, the three
        spatial dimensions, and channels.
    Returns
    -------
    names : list of string
        The name of each image series.
    sizes : list of tuple of int
        The pixel size in the specified order of each series.
    resolutions : list of tuple of float
        The resolution of each series in the order given by
        `array_order`. Time and channel dimensions are ignored.
    """
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



filename = '/mnt/shared/data/microscopy/4-SR_1_9_16hIPTG_6hM9_TS_MC5.nd2'
base_name = Path(filename).name
md = bf.get_omexml_metadata(filename)
rdr = bf.ImageReader(filename, perform_init=True)
mdroot = ETree.fromstring(md)
names, sizes, resolutions = parse_xml_metadata(md)
# We assume uniform shape + size for all of our input frames.
num_of_frames, x_dim, y_dim, channels = sizes[0]
known_good_frame_indicies = []
derp = rdr.read(t=0, series=0)
print(derp.shape)
print(derp.dtype)
for i in range(len(names)):
    output_array = []
    for j in range(num_of_frames):
        blank_check = rdr.read(c=1, t=j, series=i)
        if np.sum(blank_check) == 0:
            continue
        else:
            channel_array = []
            for k in range(channels):
                temp = rdr.read(c=k, t=j, series=i)
                channel_array.append(temp)
            output_array.append(channel_array)
    out_array = np.asarray(output_array)
    out_array = out_array.transpose(1, 0, 2, 3)
    print(out_array.shape)
    tiffile.imsave(f'{base_name}_{i}.tiff', out_array)

# thing = tiffile.imread('/mnt/shared/code/damp_lab/beholder/research/reference_notebooks/4-SR_1_9_16hIPTG_6hM9_TS_MC5.nd2_0.tiff')
# print(thing)