# Script used to collapse crops of fish classes according to bounding boxes annotated for object detection data.

import os
import xml
import cv2
import yaml

import lib.scripts.file_utils as fu

if __name__ == "__main__":
    # we expect, as a hand-shake agreement, that there is a .yml config file in top level of lib/configs directory
    config_dir = os.path.join(os.pardir, '..', 'configs', 'image_scraping')
    yaml_path = os.path.join(config_dir, 'obj_det_dataset_scraper.yml')
    with open(yaml_path, "r") as stream:
        config = yaml.load(stream)

    ## collect hyper parameters/args from config
    # NOTE: float() is required to parse any exponentials since YAML sends exponentials as strings
    src_ann_directory = config["src_ann_directory"]
    src_img_directory = config["src_img_directory"]
    dst_img_directory = config["dst_img_directory"]
    annotation_extension = config["annotation_extension"]

    classes = []
    ann_paths = fu.find_files(directory=src_ann_directory, extension=annotation_extension)

    for idx, ann_path in enumerate(ann_paths):
        et = xml.etree.ElementTree.parse(ann_path)
        root = et.getroot()

        img_filename = None
        img_path = None
        img = None
        idx = 0
        for elem in root.getiterator():
            if elem.tag == 'filename':
                # open image
                img_filename = elem.text
                img_path = os.path.join(src_img_directory, img_filename)
                img = cv2.imread(img_path)
            elif elem.tag == 'object':
                label =  elem._children[0].text
                label = label.replace(" ", "_")  # replace the spaces in multi word names with underscores since the labels get used as directory names
                if label not in classes:
                    classes.append(label)
                    # print("Found '{}' for ann_path: {}".format(label, ann_path))

                # collect bounding box coordinates
                xmin = int(elem._children[4]._children[0].text)
                ymin = int(elem._children[4]._children[1].text)
                xmax = int(elem._children[4]._children[2].text)
                ymax = int(elem._children[4]._children[3].text)

                # gather image crop
                crop = img[ymin:ymax+1, xmin:xmax+1]
                h, w = crop.shape[0:2]
                # print('Shape: {}'.format(crop.shape[0:2]))

                if h > 0 and w > 0:
                    if h < 10 or w < 10:
                        print("Warning::small crop: ({}, {}); xmin={}, ymin={}, xmax={}, ymax={}".format(img_filename, ann_path, xmin, ymin, xmax, ymax))

                    fu.init_directory(directory=os.path.join(dst_img_directory, label))
                    cv2.imwrite(os.path.join(dst_img_directory, label, img_filename[:-4]+'_crop_{}.jpg'.format(idx)), crop)
                else:
                    print("Invalid crop: ({}, {}); xmin={}, ymin={}, xmax={}, ymax={}".format(img_filename, ann_path, xmin, ymin, xmax, ymax))

                idx+=1

    print("Classes: {}".format(classes))