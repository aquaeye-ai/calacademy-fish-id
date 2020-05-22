# Script used to remove empty image/annotations file pairs in a directory.

import os
import yaml

import file_utils as fu

if __name__ == "__main__":
    # we expect, as a hand-shake agreement, that there is a .yml config file in top level of lib/configs directory
    config_dir = os.path.join(os.curdir, 'configs')
    yaml_path = os.path.join(config_dir, 'remove_empty_annotations_and_images.yml')
    with open(yaml_path, "r") as stream:
        config = yaml.load(stream)

    ## collect hyper parameters/args from config
    # NOTE: float() is required to parse any exponentials since YAML sends exponentials as strings
    directory = config["directory"]
    image_extension = config["image_extension"]
    annotation_extension = config["annotation_extension"]

    fu.remove_empty_annotations_and_images(directory=directory, image_extension=image_extension, annotation_extension=annotation_extension)