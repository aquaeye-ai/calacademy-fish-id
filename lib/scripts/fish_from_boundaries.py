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
import boundary_utils as bndry


logger = logging.getLogger(__name__)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# generate fish training data from boundaries in json
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def fish_from_boundaries(jsondata, imagedata, maskdata):
    """
    """
    lines = bndry.json_boundaries(jsondata['lines'])  # end-points and labels
    fish_boundaries = [line for line in lines if line['label'] == bndry.FISH_BOUNDARY]
    
    boundaries, intersections = bndry.determine_boundaries_and_intersections(fish_boundaries)
    slopes = bndry.find_slopes(boundaries, intersections)
    bndry.label_perimeters(slopes, boundaries, intersections)
    
    # Draw the perimenters in an image. 
    # There may be more than one if there is more than one building.
    
    fish_image = np.zeros(imagedata.shape[:2], np.uint8)
    for slope in slopes:
        poly = np.array(slope['points'], dtype=np.int32)
        if slope['perimeter']:
            # logger.info('poly: {}'.format(poly))
            cv2.fillPoly(fish_image, [poly], color=(255,255,255))
    
    return fish_image


def generate_fish_boundaries(file_tuple=None, gt_dst_dir=None, supplementals_dst_dir=None):
    """
    Call fish_from_boundaries() for tile_tuple and save output to dst_dir
    """

    jsonfile = file_tuple[1]

    #imagefile = file_tuple[0][:-4]
    imagefile = file_tuple[0]

    base_json_name = ntpath.basename(jsonfile)
    base_json_name = os.path.splitext(base_json_name)[0]
    base_json_name = base_json_name.replace('_annotation', '')
    output_file = os.path.join(gt_dst_dir, base_json_name + '.fish.png')
    
    with open(jsonfile, 'r') as f:
        jsondata = json.load(f)

    imagedata = cv2.imread(imagefile)
    fishmask = fish_from_boundaries(jsondata, imagedata, None)
    cv2.imwrite(output_file, fishmask)
    
    imagedata[:,:,1] = fishmask[:,:]  # put the mask in the green channel
    cv2.imwrite(os.path.join(supplementals_dst_dir, base_json_name + '.fish+image.png'), imagedata)

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
        fish_mask = fish_from_boundaries(json_data, image_data, None)

        return fish_mask


def main(img_src_dir=None, ann_src_dir=None, gt_dst_dir=None, supplementals_dst_dir=None):
    filenames = file_utils.find_image_json_pairs_separate_dirs(img_dir=img_src_dir, ann_dir=ann_src_dir)

    boundary_files = []

    for tuple in filenames:
        image_name = tuple[0]
        json_name = tuple[1]
        logger.info("\nprocessing: {}, {}".format(image_name, json_name))
        boundary_file = generate_fish_boundaries(file_tuple=tuple, gt_dst_dir=gt_dst_dir, supplementals_dst_dir=supplementals_dst_dir)
        boundary_files.append(boundary_file)

    cv2.destroyAllWindows()

    return boundary_files


if __name__ == "__main__":
    img_src_dir = '/home/nightrider/calacademy-fish-id/datasets/pcr/train/images/'
    ann_src_dir = '/home/nightrider/calacademy-fish-id/datasets/pcr/train/annotations'
    gt_dst_dir = '/home/nightrider/calacademy-fish-id/datasets/pcr/train/gt/'
    supplementals_dst_dir = '/home/nightrider/calacademy-fish-id/datasets/pcr/train/supplementals/'

    main(img_src_dir=img_src_dir, ann_src_dir=ann_src_dir, gt_dst_dir=gt_dst_dir, supplementals_dst_dir=supplementals_dst_dir)
    
    cv2.destroyAllWindows()