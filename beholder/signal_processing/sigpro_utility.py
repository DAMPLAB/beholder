'''
--------------------------------------------------------------------------------
Description:

Roadmap:

Written by W.R. Jackson <wrjackso@bu.edu>, DAMP Lab 2020
--------------------------------------------------------------------------------
'''
from xml.etree import ElementTree as ETree

import numpy as np
import tiffile


# --------------------------- Utility Functionality ----------------------------
def ingress_tiff_file(input_fn: str, eight_bit: bool = False):
    tiff = tiffile.imread(input_fn)
    if eight_bit:
        return tiff
    else:
        return (tiff * 65536).round().astype(np.uint16)


def get_channel_and_wl_data_from_xml_metadata(xml_tree: ETree.ElementTree):
    metadata_root = xml_tree.getroot()
    channel_list = []
    for child in metadata_root:
        inner_list = []
        if child.tag.endswith('Image'):
            for grandchild in child:
                if grandchild.tag.endswith('Pixels'):
                    for greatgrandchild in grandchild:
                        gg_attr = greatgrandchild.attrib
                        if 'EmissionWavelength' not in gg_attr:
                            continue
                        wl = gg_attr['EmissionWavelength']
                        name = gg_attr['Name']
                        inner_list.append([wl, name])
        channel_list.append(inner_list)
    # This is probably a fragile way of doing this.
    channel_list = list(filter(lambda x: len(x), channel_list))
    return channel_list


def get_channel_data_from_xml_metadata(xml_tree: ETree.ElementTree):
    metadata_root = xml_tree.getroot()
    channel_st = set()
    for child in metadata_root:
        if child.tag.endswith('Image'):
            for grandchild in child:
                if grandchild.tag.endswith('Pixels'):
                    for greatgrandchild in grandchild:
                        gg_attr = greatgrandchild.attrib
                        if 'EmissionWavelength' not in gg_attr:
                            continue
                        name = gg_attr['Name']
                        channel_st.add(name)
    return list(channel_st)


def get_time_stamps_from_xml_metadata(xml_tree: ETree.ElementTree):
    metadata_root = xml_tree.getroot()
    master_list = []
    for child in metadata_root:
        if child.tag.endswith('Image'):
            for grandchild in child:
                if grandchild.tag.endswith('Pixels'):
                    inner_list = []
                    for ggchild in grandchild:
                        if ggchild.tag.endswith('Plane'):
                            if 'DeltaT' in ggchild.attrib:
                                inner_list.append(float(ggchild.attrib['DeltaT']))
                    inner_list = set(inner_list)
                    inner_list = sorted(inner_list)
                    master_list.append(inner_list)
    return master_list


def get_channel_name_from_wavelength(wavelength_nm: float):
    wavelength_conversion_lut = {
        645.5: 'm-Cherry',
        535.0: 'GFP',
    }
    return wavelength_conversion_lut[wavelength_nm]
