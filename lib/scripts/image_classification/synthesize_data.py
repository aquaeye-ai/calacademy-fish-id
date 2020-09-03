import os
import cv2
import yaml
import random

import numpy as np
import lib.scripts.file_utils as fu

from tensorflow import keras

if __name__ == "__main__":
    # we expect, as a hand-shake agreement, that there is a .yml config file in top level of lib/configs directory
    config_dir = os.path.join('..', '..', 'configs', 'image_classification')
    yaml_path = os.path.join(config_dir, 'synthesize_data.yml')
    with open(yaml_path, "r") as stream:
        config = yaml.load(stream)

    ## collect hyper parameters/args from config
    # NOTE: float() is required to parse any exponentials since YAML sends exponentials as strings
    directory_profiles = config["directory_profiles"]
    directory_backgrounds = config["directory_backgrounds"]
    directory_output = config["directory_output"]
    num_images = config["num_images"]
    profile_preprocessing = config["profile_preprocessing"]
    background_preprocessing = config["background_preprocessing"]

    background_images = fu.find_images(directory=directory_backgrounds, extension='.jpg')

    # synthetically create data for each class
    class_dirs = [d for d in os.listdir(directory_profiles)]
    for class_dir in class_dirs:
        class_dir_path = os.path.join(directory_profiles, class_dir)
        profile_images = fu.find_images(directory=class_dir_path, extension='.png')

        # create output directory for class
        fu.init_directory(directory=os.path.join(directory_output, class_dir))

        # loop until we have created num_images images
        for i in range(num_images):
            # randomly choose profile
            rand_img_profile = random.choice(profile_images)
            print("rand_profile_image: {}".format(rand_img_profile))

            # read in profile image
            img_profile = cv2.imread(rand_img_profile)

            # cv2.imshow('original profile', img_profile)
            # cv2.waitKey(0)

            ## Apply preprocessing to profile

            # horizontal and vertical flip
            if  profile_preprocessing['horizontal_flip'] > 0:
                img_profile = cv2.flip(img_profile, 1)

            # cv2.imshow('horizontally flipped profile', img_profile)
            # cv2.waitKey(0)

            if profile_preprocessing['vertical_flip'] > 0:
                print('here')
                img_profile = cv2.flip(img_profile, 0)

            cv2.imshow('vertically flipped profile', img_profile)
            cv2.waitKey(0)

            # rotation

            # grab image center
            h, w = img_profile.shape[:2]
            cX, cY = (w // 2, h // 2)

            # generate random angle
            rot_angle = random.uniform(profile_preprocessing['rotation_range'][0],
                                       profile_preprocessing['rotation_range'][1])

            # grab the rotation matrix (applying the negative of the
            # angle to rotate clockwise), then grab the sine and cosine
            # (i.e., the rotation components of the matrix)
            M = cv2.getRotationMatrix2D((cX, cY), -rot_angle, 1.0)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])

            # compute the new bounding dimensions of the image
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))

            # adjust the rotation matrix to take into account translation
            M[0, 2] += (nW / 2) - cX
            M[1, 2] += (nH / 2) - cY

            # perform rotation
            img_profile = cv2.cv2.warpAffine(img_profile, M, (nW, nH))

            cv2.imshow('rotated profile', img_profile)
            cv2.waitKey(0)