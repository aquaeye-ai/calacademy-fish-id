"""
Script used to collapse set of classes in csv to a subset of those classes.
Used after xml_to_csv_pbtxt.py but before to_TFRecord.py when generating object detection data.
"""


import os
import yaml

import lib.scripts.file_utils as fu

if __name__ == "__main__":
    # we expect, as a hand-shake agreement, that there is a .yml config file in top level of lib/configs directory
    config_dir = os.path.join('..', '..', 'configs', 'object_detection')
    yaml_path = os.path.join(config_dir, 'collapse_classes_to_subset_in_csv.yml')
    with open(yaml_path, "r") as stream:
        config = yaml.load(stream)

    ## collect hyper parameters/args from config
    # NOTE: float() is required to parse any exponentials since YAML sends exponentials as strings
    csv_path = config["csv_path"]
    keep_classes = config["keep_classes"]
    default_class = config["default_class"]

    fu.collapse_classes_to_subset_in_csv(csv_path=csv_path, keep_classes=keep_classes, default_class=default_class)