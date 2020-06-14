# Script used to collapse crops of fish classes according to bounding boxes annotated for object detection data.

import os
import xml
import yaml

import lib.file_utils as fu

if __name__ == "__main__":
    # we expect, as a hand-shake agreement, that there is a .yml config file in top level of lib/configs directory
    config_dir = os.path.join(os.curdir, 'configs')
    yaml_path = os.path.join(config_dir, 'scrape_obj_det_dataset.yml')
    with open(yaml_path, "r") as stream:
        config = yaml.load(stream)

    ## collect hyper parameters/args from config
    # NOTE: float() is required to parse any exponentials since YAML sends exponentials as strings
    directory = config["directory"]
    annotation_extension = config["annotation_extension"]

    classes = []
    ann_paths = fu.find_files(directory=directory, extension=annotation_extension)

    for idx, ann_path in enumerate(ann_paths):
        et = xml.etree.ElementTree.parse(ann_path)
        root = et.getroot()

        for elem in root.getiterator():
            if elem.tag == 'object':

                classes.append(elem._children[0].text)
                print("Found '{}' for ann_path: {}".format(elem._children[0].text, ann_path))