#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function


# standard libs
import os
import cv2
import json
import ntpath
import logging

import numpy as np

# personal libs
import file_utils
import viz_utils

import lib.globals as globals
import boundary_utils as bndry

from lib.file_utils import init_directory

logger = logging.getLogger(__name__)

if not globals.initialized:  # if this is called as module from another file we don't want to potentially reset the globals
    # that file set
    globals.classifier = "fish_segmentation"
    globals.model = "roof_indicator_discrete_orientation"
    globals.training_masks_dir = init_directory("../outputs/training_masks/classifiers/{}/models/{}/"
                                                .format(globals.classifier, globals.model), use_logger=False)
    globals.data_visuals_dir = init_directory("../outputs/data_visuals/classifiers/{}/models/{}/"
                                              .format(globals.classifier, globals.model), use_logger=False)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# generate fish training data from boundaries in json
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def fish_from_boundaries(jsondata, imagedata, maskdata):
    """
    """
    lines = bndry.json_boundaries(jsondata['lines'])  # end-points and labels
    fish_edges = [line for line in lines if line['label'] == bndry.FISH_BOUNDARY]

    boundaries, intersections = bndry.determine_boundaries_and_intersections(fish_edges)
    slopes = bndry.find_slopes(boundaries, intersections)
    bndry.label_perimeters(slopes, boundaries, intersections)

    # Draw the perimeters in an image.
    # There may be more than one if there is more than one fish.

    fish_image = np.zeros(imagedata.shape[:2], np.uint8)
    for slope in slopes:
        poly = np.array(slope['points'], dtype=np.int32)
        if slope['perimeter']:
            # logger.info('poly: {}'.format(poly))
            cv2.fillPoly(fish_image, [poly], color=(255, 255, 255))

    return fish_image


def generate_fish_boundaries(file_tuple=None, dst_dir=None):
    """
    Call fish_from_boundaries() for tile_tuple and save output to dst_dir
    """

    jsonfile = file_tuple[1]

    # imagefile = file_tuple[0][:-4]
    imagefile = file_tuple[0]

    base_json_name = ntpath.basename(jsonfile)
    base_json_name = os.path.splitext(base_json_name)[0]
    output_file = dst_dir + '/' + base_json_name + '.fish.png'

    with open(jsonfile, 'r') as f:
        jsondata = json.load(f)

    imagedata = cv2.imread(imagefile)
    roofmask = fish_from_boundaries(jsondata, imagedata, None)
    cv2.imwrite(output_file, roofmask)

    imagedata[:, :, 1] = roofmask[:, :]  # put the mask in the green channel
    cv2.imwrite(dst_dir + '/' + base_json_name + '.fish+image.png', imagedata)

    return output_file


def test(file_tuple=None):
    """
    Test the basic call to fish_from_boundaries()
    """

    json_file = file_tuple[1]
    image_file = file_tuple[0]

    with open(json_file, 'r') as f:
        json_data = json.load(f)

        image_data = cv2.imread(image_file)
        roof_mask = fish_from_boundaries(json_data, image_data, None)

        return roof_mask


def main(src_dir=None, dst_dir=globals.training_masks_dir):
    filenames = file_utils.find_image_json_pairs(directory=src_dir)

    boundary_files = []

    for tuple in filenames:
        image_name = tuple[0]
        json_name = tuple[1]
        logger.info("\nprocessing: {}, {}".format(image_name, json_name))
        boundary_file = generate_fish_boundaries(file_tuple=tuple, dst_dir=dst_dir)
        boundary_files.append(boundary_file)

    cv2.destroyAllWindows()

    return boundary_files


if __name__ == "__main__":

    directory = '/home/nightrider/calacademy-fish-id/datasets/pcr/train/'

    filenames1 = file_utils.find_image_json_pairs(directory=directory)

    # filenames2 = [
    #     'NC-Huntersville-DesotaLn-#-2017-02-11.png.json',  # roof OK #
    #     'MA-Walpole-Unknown-1-time-1.png.json',
    #     'MA-Walpole-Unknown-1-time-2.png.json',            # unsure, focus
    #     'TX-TheColony-PhelpsSt-5649.png.json',             # roof OK # topdown OK
    #     'MA-Walpole-Unknown-2-time-1.png.json',            # error
    # ]

    for tuple in filenames1:
        image_name = tuple[0]
        json_name = tuple[1]
        print("\nprocessing: {}, {}".format(image_name, json_name))
        # test(directory + fname)
        viz_utils.verify_line_labels(imagefile=image_name, jsonfile=json_name)
        roof_map = test(file_tuple=tuple)

        if image_name == "/home/nightrider/calacademy-fish-id/datasets/pcr/train/achilles_tang/1.png":
            cv2.imshow('image', roof_map)
            cv2.waitKey(5000)

    cv2.destroyAllWindows()