"""
Groups reef_lagoon data under datasets/image_classification/reef_lagoon.
For example, can be given the output of lib/image_scraping/obj_det_dataset_scraper.py, a mapping defined in this script ->
will then group the data from obj_det_dataset_scraper.py according to the mapping and store in a given directory.
"""


import os
import yaml
import shutil

import lib.file_utils as fu


SCIENTIFIC_SPECIES_NAMES_TO_COMMON_GROUP_NAMES = "SCIENTIFIC_SPECIES_NAMES_TO_COMMON_GROUP_NAMES"
OBJECT_DETECTION_TO_COMMON_GROUP_NAMES = "OBJECT_DETECTION_TO_COMMON_GROUP_NAMES"

DB_MAPS = {
    SCIENTIFIC_SPECIES_NAMES_TO_COMMON_GROUP_NAMES: {
        "surgeonfishes": ["acanthurus_triostegus"],
        "butterflyfishes": ["chelmon_rostratus"],
        "pompanos": ["trachinotus_mookalee"],
        "moonyfishes": ["monodactylus_argenteus"],
        "stingrays": ["himantura_uarnak", "neotrygon_kuhlii", "rhinoptera_javanica", "taeniura_lymma"]
    },
    OBJECT_DETECTION_TO_COMMON_GROUP_NAMES: {
        "surgeonfishes": ["acanthurus_triostegus", "surgeonfish"],
        "butterflyfishes": ["chelmon_rostratus", "butterflyfish"],
        "pompanos": ["trachinotus_mookalee", "pompano"],
        "moonyfishes": ["monodactylus_argenteus", "moonyfish"],
        "stingrays": ["himantura_uarnak", "neotrygon_kuhlii", "rhinoptera_javanica", "taeniura_lymma", "stingray"]
    }
}


if __name__ == "__main__":
    # we expect, as a hand-shake agreement, that there is a .yml config file in top level of lib/configs directory
    config_dir = os.path.join(os.pardir, 'configs', 'image_classification')
    yaml_path = os.path.join(config_dir, 'group_reef_lagoon_data.yml')
    with open(yaml_path, "r") as stream:
        config = yaml.load(stream)

    ## collect hyper parameters/args from config
    # NOTE: float() is required to parse any exponentials since YAML sends exponentials as strings
    directory_src = config["directory_src"]
    directory_dst = config["directory_dst"]
    db_map_key = config["db_map_key"]

    # initialize destination directory in case it doesn't exist
    fu.init_directory(directory_dst)

    # get database mapping
    db_map = DB_MAPS[db_map_key]

    # get class directories
    class_dirs = [d for d in os.listdir(directory_src)]

    # map the class directories according to the database mapping
    total = 0
    for class_dir in class_dirs:
        class_dir_path = os.path.join(directory_src, class_dir)

        # some datasets will have an 'other' class which will be empty in the mapping (e.g. when scraping object
        # detection data), so we just take 'other' as the bucket name
        bucket = None
        if class_dir != "other":
            for key in db_map.keys():
                if class_dir in db_map[key]:
                    bucket = key
        else:
            bucket = "other"

        # initialize directory to hold bucket category
        bucket_dir_path = os.path.join(directory_dst, bucket)
        fu.init_directory(bucket_dir_path)

        # move files to bucket
        num_class = 0
        for idx, f_path_src in enumerate([os.path.join(class_dir_path, f) for f in os.listdir(class_dir_path) if os.path.isfile(os.path.join(class_dir_path, f))]):
            basename = os.path.basename(f_path_src)
            f_path_dst = os.path.join(bucket_dir_path, "{}_{}".format(class_dir, basename)) # need some string in addition to basename to avoid possible collisions in names, e.g. "0.jpg" from one species folder may collide with "0.jpg" from another species folder
            shutil.copyfile(f_path_src, f_path_dst)
            total += 1
            num_class += 1

        print("{}: {}".format(class_dir, num_class))

    print("Total: {}".format(total))