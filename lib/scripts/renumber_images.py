# Script used to renumber images in a given directory.

import os
import yaml

import file_utils as fu

if __name__ == "__main__":
    # we expect, as a hand-shake agreement, that there is a .yml config file in top level of lib/configs directory
    config_dir = os.path.join(os.curdir, 'configs')
    yaml_path = os.path.join(config_dir, 'renumber_images.yml')
    with open(yaml_path, "r") as stream:
        config = yaml.load(stream)

    ## collect hyper parameters/args from config
    # NOTE: float() is required to parse any exponentials since YAML sends exponentials as strings
    source_directory = config["source_directory"]
    destination_directory = config["destination_directory"]
    start_number = config["start_number"]
    image_extension = config["image_extension"]

    fu.renumber_images(source_directory=source_directory, destination_directory=destination_directory,
                                               start_number=start_number, image_extension=image_extension)