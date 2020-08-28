# Script used to rename images' extensions in a given directory.
# Reduces image extensions to either .png or .jpg based on the presence of either in the extension.
# E.g. '<image_name>.jpg&w=600' -> '<image_name>.jpg' or '<image_name>.pngblaghgh12w' -> '<image_name>.png'
#
# Useful for cleaning the image extensions of scraped Bing/Google images (especially Bing) since some of their images
# tend to have strange file extensions.

import os
import yaml

import file_utils as fu

if __name__ == "__main__":
    # we expect, as a hand-shake agreement, that there is a .yml config file in top level of lib/configs directory
    config_dir = os.path.join(os.curdir, 'configs')
    yaml_path = os.path.join(config_dir, 'clean_image_extensions.yml')
    with open(yaml_path, "r") as stream:
        config = yaml.load(stream)

    ## collect hyper parameters/args from config
    # NOTE: float() is required to parse any exponentials since YAML sends exponentials as strings
    source_directory = config["source_directory"]
    destination_directory = config["destination_directory"]

    fu.clean_image_extensions(source_directory=source_directory, destination_directory=destination_directory)