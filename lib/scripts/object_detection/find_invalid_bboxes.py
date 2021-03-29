# Script used to find empty and near-empty bounding boxes annotated for object detection data.
# Saves the found image/bounding-box pairs to a given directory for later inspection.

import os
import xml
import cv2
import yaml
import shutil

import lib.scripts.file_utils as fu

if __name__ == "__main__":
    # we expect, as a hand-shake agreement, that there is a .yml config file in top level of lib/configs directory
    config_dir = os.path.join('..', '..', 'configs', 'object_detection')
    yaml_path = os.path.join(config_dir, 'find_invalid_bboxes.yml')
    with open(yaml_path, "r") as stream:
        config = yaml.load(stream)

    ## collect hyper parameters/args from config
    # NOTE: float() is required to parse any exponentials since YAML sends exponentials as strings
    src_ann_directory = config["src_ann_directory"]
    src_img_directory = config["src_img_directory"]
    dst_directory = config["dst_directory"]
    annotation_extension = config["annotation_extension"]
    keep_invalid_boxes = config["keep_invalid_boxes"]

    fu.init_directory(directory=dst_directory)

    classes = []
    ann_paths = fu.find_files(directory=src_ann_directory, extension=annotation_extension)

    for idx, ann_path in enumerate(ann_paths):
        et = xml.etree.ElementTree.parse(ann_path)
        root = et.getroot()

        img_filename = None
        img_path = None
        idx = 0
        h_img, w_img = None, None
        for elem in root.getiterator():
            if elem.tag == 'filename':
                # gather image path and name
                img_filename = elem.text + '.jpg'
                img_path = os.path.join(src_img_directory, img_filename)
            elif elem.tag == 'size':
                # gather image dimensions
                w_img = int(elem._children[0].text)
                h_img = int(elem._children[1].text)
            elif elem.tag == 'object':
                label =  elem._children[0].text
                if label not in classes:
                    classes.append(label)
                    # print("Found '{}' for ann_path: {}".format(label, ann_path))

                # collect bounding box coordinates
                xmin = int(elem._children[4]._children[0].text)
                ymin = int(elem._children[4]._children[1].text)
                xmax = int(elem._children[4]._children[2].text)
                ymax = int(elem._children[4]._children[3].text)

                # crop could be at edge of image in which case the height or width would be off by one
                h_crop = ymax - ymin + 1 if ymax < h_img else ymax - ymin
                w_crop = xmax - xmin + 1 if xmax < w_img else xmax - xmin

                if (h_crop > 0 and w_crop > 0) and (h_crop < 10 or w_crop < 10):
                    print("Warning::Small Crop Detected: ({}, {}); shape={}x{}, xmin={}, ymin={}, xmax={}, ymax={}".format(img_filename, ann_path, h_crop, w_crop, xmin, ymin, xmax, ymax))

                    # copy image to dst_directory for manual inspection, e.g. using LabelImg
                    shutil.copyfile(img_path, os.path.join(dst_directory, img_filename))

                    if keep_invalid_boxes:
                        # copy annotation to dst_directory for manual inspection, e.g. using LabelImg
                        shutil.copyfile(ann_path, os.path.join(dst_directory, img_filename[:-3]+annotation_extension))
                    else:
                        # remove invalid boxes from annotation and save to dst_directory
                        root.remove(elem)
                        et.write(os.path.join(dst_directory, img_filename[:-3]+annotation_extension))

                idx+=1

    print("Classes: {}".format(classes))