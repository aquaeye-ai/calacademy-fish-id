"""
Builds train/val/test splits from two datasets, one given from /datasets/image_classification/reef_lagoon/scraped_obj_det and
one from /datasets/image_classification/reef_lagoon/scraped_web.  These two datasets' classes don't have to intersect.
"""

import os
import yaml
import random
import shutil

import numpy as np
import lib.file_utils as fu


def partition_data(directory_src=None, class_dirs=None, type=None, percent_train=None, percent_val=None, percent_test=None):
    print("Type: {}".format(type))

    total_train = 0
    total_val = 0
    total_test = 0

    # partition web data
    for class_dir in class_dirs:
        # Create class directory in case it doesn't exist yet
        class_dir_train = os.path.join(train_dir, class_dir)
        class_dir_val = os.path.join(val_dir, class_dir)
        class_dir_test = os.path.join(test_dir, class_dir)

        fu.init_directory(class_dir_train)
        fu.init_directory(class_dir_val)
        fu.init_directory(class_dir_test)

        # get number for splits
        class_dir_src = os.path.join(directory_src, class_dir)
        class_images = fu.find_images(directory=class_dir_src, extension='.jpg')
        num_class = len(class_images)

        # if there are fewer than 10 images in the class, we simply put them all into test since it's difficult to
        # make splits of less than 10 images due to rounding
        if num_class < 10:
            num_class_train = 0
            num_class_val = 0
            num_class_test = num_class
        else:
            num_class_val = int(np.floor((np.float(percent_val) / 100) * num_class))
            num_class_test = int(np.floor((np.float(percent_test) / 100) * num_class))

            # rounding causes us to sometimes use less than num_classes so we just throw the remaining images into train so as to utilize all images
            num_class_train = num_class - num_class_val - num_class_test

        # randomly gather images for splits
        rand_class_val_ins = sorted(random.sample(range(num_class), num_class_val))
        rand_class_test_ins = sorted(random.sample(range(num_class), num_class_test))
        rand_class_train_ins = sorted(random.sample(range(num_class), num_class_train))

        class_images_train = [class_images[i] for i in rand_class_train_ins]
        class_images_val = [class_images[i] for i in rand_class_val_ins]
        class_images_test = [class_images[i] for i in rand_class_test_ins]

        # copy files to destination splits
        for img in class_images_train:
            basename = os.path.basename(img)[:-4]
            shutil.copy(img, os.path.join(class_dir_train, "{}_{}.jpg".format(basename, type)))

        for img in class_images_val:
            basename = os.path.basename(img)[:-4]
            shutil.copy(img, os.path.join(class_dir_val, "{}_{}.jpg".format(basename, type)))

        for img in class_images_test:
            basename = os.path.basename(img)[:-4]
            shutil.copy(img, os.path.join(class_dir_test, "{}_{}.jpg".format(basename, type)))

        total_train += num_class_train
        total_val += num_class_val
        total_test += num_class_test

        assert((num_class_train + num_class_val + num_class_test) == num_class)

        print("train: {}, {}".format(class_dir, num_class_train))
        print("val: {}, {}".format(class_dir, num_class_val))
        print("test: {}, {}\n".format(class_dir, num_class_test))

    return total_train, total_val, total_test


if __name__ == "__main__":
    # we expect, as a hand-shake agreement, that there is a .yml config file in top level of lib/configs directory
    config_dir = os.path.join(os.pardir, 'configs', 'image_classification')
    yaml_path = os.path.join(config_dir, 'build_reef_lagoon_data_splits.yml')
    with open(yaml_path, "r") as stream:
        config = yaml.load(stream)

    ## collect hyper parameters/args from config
    # NOTE: float() is required to parse any exponentials since YAML sends exponentials as strings
    directory_web = config["directory_web"]
    directory_od = config["directory_od"]
    directory_dst = config["directory_dst"]
    percent_train = config["percent_train"]
    percent_val = config["percent_val"]
    percent_test = config["percent_test"]

    # initialize destination directories for data splits
    train_dir = os.path.join(directory_dst, "train")
    val_dir = os.path.join(directory_dst, "val")
    test_dir = os.path.join(directory_dst, "test")

    fu.init_directory(train_dir)
    fu.init_directory(val_dir)
    fu.init_directory(test_dir)

    # get class directories and partition data
    # we may not be using both web and obj det data, so check first
    web_total_train, web_total_val, web_total_test = 0, 0, 0
    if directory_web is not None:
        class_dirs_web = [d for d in os.listdir(directory_web)]

        web_total_train, web_total_val, web_total_test = partition_data(directory_src=directory_web,
                                                                        class_dirs=class_dirs_web,
                                                                        type="web",
                                                                        percent_train=percent_train,
                                                                        percent_val=percent_val,
                                                                        percent_test=percent_test)

    od_total_train, od_total_val, od_total_test = 0, 0, 0
    if directory_od is not None:
        class_dirs_od = [d for d in os.listdir(directory_od)]

        od_total_train, od_total_val, od_total_test = partition_data(directory_src=directory_od,
                                                                     class_dirs=class_dirs_od,
                                                                     type="od",
                                                                     percent_train=percent_train,
                                                                     percent_val=percent_val,
                                                                     percent_test=percent_test)

    # copy config so we can recall what parameters were used to construct the dataset splits
    shutil.copy(yaml_path, os.path.join(directory_dst, 'config.yml'))

    print("\nWeb Totals: train_{}, val_{}, test_{}".format(web_total_train, web_total_val, web_total_test))
    print("OD Totals: train_{}, val_{}, test_{}".format(od_total_train, od_total_val, od_total_test))