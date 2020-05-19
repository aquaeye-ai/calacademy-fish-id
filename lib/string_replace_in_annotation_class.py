# Script used to rename an annotation class throughout a directory.

import os
import yaml

import file_utils as fu

if __name__ == "__main__":
    # we expect, as a hand-shake agreement, that there is a .yml config file in top level of lib/configs directory
    config_dir = os.path.join(os.curdir, 'configs')
    yaml_path = os.path.join(config_dir, 'rename_annotation_class.yml')
    with open(yaml_path, "r") as stream:
        config = yaml.load(stream)

    ## collect hyper parameters/args from config
    # NOTE: float() is required to parse any exponentials since YAML sends exponentials as strings
    directory = config["directory"]
    old_classes = config["old_classes"]
    new_classes = config["new_classes"]
    annotation_extension = config["annotation_extension"]

    #TODO: This could be done more efficiently with only one iteration through the dataset for all classes
    for ptuple in zip(old_classes, new_classes):
        fu.rename_annotation_class(directory=directory, old_class=ptuple[0], new_class=ptuple[1], annotation_extension=annotation_extension)