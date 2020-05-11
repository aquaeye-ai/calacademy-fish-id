# Script to use the tiling functions for already-annotated images provided here: https://github.com/jdcast/image_bbox_slicer

import os
import yaml

# pycharm config for this script sets the interpreter to one in an virtual environment that has this package installed,
# so we can ignore the this import error
import image_bbox_slicer as ibs


if __name__ == "__main__":
    # we expect, as a hand-shake agreement, that there is a .yml config file in top level of lib/configs directory
    config_dir = os.path.join(os.curdir, 'configs')
    yaml_path = os.path.join(config_dir, 'tile_annotated_images.yml')
    with open(yaml_path, "r") as stream:
        config = yaml.load(stream)

    ## collect hyper parameters/args from config
    # NOTE: float() is required to parse any exponentials since YAML sends exponentials as strings
    save_before_after_map = config["save_before_after_map"]
    keep_partial_labels = config["keep_partial_labels"]
    tile_overlap = config["tile_overlap"]
    model = config["model"]
    models_config = config["models_config"]
    im_src = config["im_src"]
    an_src = config["an_src"]
    im_dst = config["im_dst"]
    an_dst = config["an_dst"]

    slicer = ibs.Slicer()
    slicer.config_dirs(img_src=im_src,
                       ann_src=an_src,
                       img_dst=im_dst,
                       ann_dst=an_dst)

    h = models_config[model]['image_size']['height']
    w = models_config[model]['image_size']['width']
    slicer.save_before_after_map = True if save_before_after_map > 0 else False
    slicer.keep_partial_labels = True if keep_partial_labels > 0 else False
    slicer.slice_by_size(tile_size=(h, w), tile_overlap=tile_overlap)

    # Can't use this line if script is executed within pycharm because of installation bug for tkinter so in this case
    # we can instead use labelImg to visualize the annotations on the images.
    # However, this line will work if the script is run outside of pycharm with python 2.7.
    slicer.visualize_sliced_random()