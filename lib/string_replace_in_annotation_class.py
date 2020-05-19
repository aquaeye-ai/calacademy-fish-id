# Script used to rename an annotation class throughout a directory.

import os
import yaml

import file_utils as fu

if __name__ == "__main__":
    # we expect, as a hand-shake agreement, that there is a .yml config file in top level of lib/configs directory
    config_dir = os.path.join(os.curdir, 'configs')
    yaml_path = os.path.join(config_dir, 'string_replace_in_annotation_class.yml')
    with open(yaml_path, "r") as stream:
        config = yaml.load(stream)

    ## collect hyper parameters/args from config
    # NOTE: float() is required to parse any exponentials since YAML sends exponentials as strings
    directory = config["directory"]
    old_str = config["old_str"]
    new_str = config["new_str"]
    annotation_extension = config["annotation_extension"]

    fu.string_replace_in_annotation_class(directory=directory, old_str=old_str, new_str=new_str, annotation_extension=annotation_extension)