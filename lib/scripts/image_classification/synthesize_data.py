"""
Script that inserts a profile (png) into a crop taken from a background (jpg).
Applies augmentation to both profile and background.
Gives profile some random padding before inserting into a background crop.
The alpha channel of the profile is necessary and is used to blend the profile and background (link for further reading
in relevant code below).
"""

import os
import cv2
import yaml
import math
import shutil
import random

import numpy as np
import lib.scripts.file_utils as fu


def preprocess_image(image_np=None, preprocessing_dict=None):
    """
    Applies the preprocessing defined in preprocessing_dict to an image.

    :param image_np: np array, image to preprocess
    :param preprocessing_dict: dictionary, preprocessing to apply
    :return: np arrary, preprocessed image
    """
    ## horizontal and vertical flip
    if preprocessing_dict['horizontal_flip'] > 0:
        if random.random() > 0.5:
            image_np = cv2.flip(image_np, 1)

    # cv2.imshow('horizontally flipped', img_profile)
    # cv2.waitKey(0)

    if preprocessing_dict['vertical_flip'] > 0:
        if random.random() > 0.5:
            image_np = cv2.flip(image_np, 0)

    # cv2.imshow('vertically flipped', image_np)
    # cv2.waitKey(0)

    ## brightness and contrast

    brightness = 0
    if preprocessing_dict['brightness']['apply'] > 0:
        brightness = random.uniform(preprocessing_dict['brightness']['brightness_range'][0],
                                    preprocessing_dict['brightness']['brightness_range'][1])
    contrast = 1
    if preprocessing_dict['contrast']['apply'] > 0:
        contrast = random.uniform(preprocessing_dict['contrast']['contrast_range'][0],
                                  preprocessing_dict['contrast']['contrast_range'][1])

    # apply brightness and contrast according to: https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
    # NOTE: this is slow
    if preprocessing_dict['brightness']['apply'] > 0 or preprocessing_dict['contrast']['apply'] > 0:
        for y in range(image_np.shape[0]):
            for x in range(image_np.shape[1]):
                for c in range(image_np.shape[2] - 1): # don't overwrite the alpha channel, so only modify RGB channels
                    image_np[y, x, c] = np.clip(contrast * image_np[y, x, c] + brightness, 0, 255)

    # cv2.imshow('brightness + contrast', image_np)
    # cv2.waitKey(0)

    ## shear

    if preprocessing_dict['shear']['apply'] > 0:
        h, w = image_np.shape[:2]

        # choose random shear for x and y
        shear_x = random.uniform(preprocessing_dict['shear']['shear_range'][0],
                                 preprocessing_dict['shear']['shear_range'][1])
        shear_y = random.uniform(preprocessing_dict['shear']['shear_range'][0],
                                 preprocessing_dict['shear']['shear_range'][1])

        new_cols = shear_x * w
        new_rows = shear_y * h

        # TODO: remove; shouldn't need these according to this: https://stackoverflow.com/questions/44425701/shear-an-image-without-cropping and https://stackoverflow.com/questions/57881430/how-could-i-implement-a-centered-shear-an-image-with-opencv
        h_pad = int(math.ceil(shear_y*w/2))
        w_pad = int(math.ceil(shear_x*h/2))

        up_down = int(math.ceil(new_rows / 2)) + h_pad
        left_right = int(math.ceil(new_cols / 2)) + w_pad

        image_np = cv2.copyMakeBorder(image_np, up_down, up_down, left_right, left_right, cv2.BORDER_CONSTANT, value=(255, 255, 255, 0))
        rows, cols, ch = image_np.shape

        # both x and y shear
        translat_center_x = -new_cols / 2 - (shear_x*h/2)
        translat_center_y = -new_rows / 2 - (shear_y*w/2)

        M = np.float64([[1, shear_x, translat_center_x], [shear_y, 1, translat_center_y]])
        image_np = cv2.warpAffine(image_np, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255, 0))

        # cv2.imshow('shear', image_np)
        # cv2.waitKey(0)

    ## rotation

    # rotate after shear since we noticed that otherwise, many of the fish profiles seem slanted in the same direction
    # NOTE: it may be possible to combine rotation with shear matrix as described here: https://stackoverflow.com/questions/44425701/shear-an-image-without-cropping
    # however, we don't combine here to maintain clarity of code
    if preprocessing_dict['rotation']['apply'] > 0:
        # generate random angle
        angle = random.uniform(preprocessing_dict['rotation']['rotation_range'][0],
                               preprocessing_dict['rotation']['rotation_range'][1])
        image_np = rotate_image(image_np=image_np, angle=angle)

    # cv2.imshow('rotated', image_np)
    # cv2.waitKey(0)

    return image_np

def rotate_image(image_np=None, angle=None):
    """
    Rotates image with a random angle we avoid cut-off as described here: https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/)

    :param image_np: np array, image to rotate
    :return: np array, preprocessed image
    """
    # grab image center
    h, w = image_np.shape[:2]
    cX, cY = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform rotation
    image_np = cv2.cv2.warpAffine(image_np, M, (nW, nH))

    # cv2.imshow('rotated', image_np)
    # cv2.waitKey(0)

    return image_np

def combine_profile_and_mask(img_profile=None, img_background=None, background_preprocessing_dict=None):
    """
    Randomly selects a crop of a background image.  Then selects a random margin to give the crop so that the crop will
    fit a profile image with some margin.  Then combines the profile and background crop together using alpha channel
    that should be present within the profile image.  Works best if profile image is a png.

    :param img_profile: np array, image of profile to paste onto background crop, should have a fourth channel corresponding
    to alpha
    :param img_background: np array, image of background from which to randomly select crop
    :return: np array, final image of profile pasted onto a crop of background (does not contain an alpha channel) or
    None if profile/background combination were incompatible due to size differences
    """
    # generate random margin to give profile as percentage of its height/width
    h_profile, w_profile = img_profile.shape[:2]
    h_background, w_background = img_background.shape[0:2]

    # check to make sure profile will fit into background
    # if it does then continue, otherwise skip
    res = None
    if h_profile <= h_background and w_profile <= w_background:
        rand_margin_percent = random.uniform(margin_range[0], margin_range[1])

        # convert the margin from percentage to pixel value for height/width dimensions

        # find the maximum margin we can use
        max_allowable_margin_pixel_h = h_background - h_profile
        max_allowable_margin_pixel_w = w_background - w_profile

        # we need to take a min of our chosen margin and max margin so that the margin doesn't make the new profile larger
        # than the dimensions of the background
        rand_margin_pixel_h = min([int((rand_margin_percent / 100) * h_profile), max_allowable_margin_pixel_h])
        rand_margin_pixel_w = min([int((rand_margin_percent / 100) * w_profile), max_allowable_margin_pixel_w])

        # gather new height/width based on margin to give profile
        h_profile_new = h_profile + rand_margin_pixel_h
        w_profile_new = w_profile + rand_margin_pixel_w

        # determine allowable range in height/width, relative to center of background image, to randomly
        # choose crop from the background image
        h_diff = h_background - h_profile_new
        w_diff = w_background - w_profile_new
        cX_range_offset, cY_range_offset = (w_diff // 2, h_diff // 2)

        # choose random center for crop
        cX_rand_relative, cY_rand_relative = (
        int(random.uniform(-cX_range_offset, cX_range_offset)), int(random.uniform(-cY_range_offset, cY_range_offset)))

        # convert the chosen center to pixel values relative to the height/width of the background image
        cX_background, cY_background = (w_background // 2, h_background // 2)
        cX_rand_absolute, cY_rand_absolute = (cX_background + cX_rand_relative, cY_background + cY_rand_relative)

        h_crop_start = cY_rand_absolute - (h_profile_new // 2)
        h_crop_end = cY_rand_absolute + (h_profile_new // 2)
        w_crop_start = cX_rand_absolute - (w_profile_new // 2)
        w_crop_end = cX_rand_absolute + (w_profile_new // 2)

        # adjust the start and end indices in case the integer rounding made their range too small to fit profile plus its margin
        if h_crop_end - h_crop_start < h_profile_new:
            # check if this resides out of bounds for the background
            if h_crop_end + 1 > h_background:
                h_crop_start -= 1
            else:
                h_crop_end += 1

        if w_crop_end - w_crop_start < w_profile_new:
            # check if this resides out of bounds for the background
            if w_crop_end + 1 > w_background:
                w_crop_start -= 1
            else:
                w_crop_end += 1

        # extract the crop
        crop = img_background[h_crop_start:h_crop_end, w_crop_start:w_crop_end]
        # cv2.imshow('random_background_crop', crop)
        # cv2.waitKey(0)

        # preprocess background crop
        # we preprocess only the crop and not the entire background because otherwise the preprocessing is too slow
        crop = preprocess_image(image_np=crop,
                                preprocessing_dict=background_preprocessing_dict)

        ## paste profile onto background crop using alpha blending as described here: https://www.learnopencv.com/alpha-blending-using-opencv-cpp-python/

        # resize profile to be same dimensions as the crop
        img_profile_resized = np.zeros(
            (crop.shape[0], crop.shape[1], img_profile.shape[2]))  # make sure to grab alpha channel

        # center profile into new profile
        h_profile_resized, w_profile_resized = img_profile_resized.shape[:2]
        cX_profile_resized, cY_profile_resized = (w_profile_resized // 2, h_profile_resized // 2)
        h_profile_resized_start = cY_profile_resized - (h_profile // 2)
        h_profile_resized_end = cY_profile_resized + (h_profile // 2)
        w_profile_resized_start = cX_profile_resized - (w_profile // 2)
        w_profile_resized_end = cX_profile_resized + (w_profile // 2)

        # adjust the start and end indices in case the integer rounding made their range too small to fit alpha
        if h_profile_resized_end - h_profile_resized_start < h_profile:
            # check if this resides out of bounds for the crop
            if h_profile_resized_end + 1 > h_profile_resized:
                h_profile_resized_start -= 1
            else:
                h_profile_resized_end += 1

        if w_profile_resized_end - w_profile_resized_start < w_profile:
            # check if this resides out of bounds for the crop
            if w_profile_resized_end + 1 > w_profile_resized:
                w_profile_resized_start -= 1
            else:
                w_profile_resized_end += 1

        img_profile_resized[
        h_profile_resized_start:h_profile_resized_end,
        w_profile_resized_start:w_profile_resized_end
        ] = img_profile

        # grab alpha mask
        alpha = img_profile_resized[:, :, 3]
        alpha = np.repeat(alpha[:, :, np.newaxis], 3,
                          axis=2)  # we need to keep alpha, img_profile and crop of the same dimensions

        # convert uint8 to float
        crop = crop.astype(float)
        img_profile_resized = img_profile_resized.astype(float)[:, :, :3]  # we don't need the alpha channel any longer

        # normalize the alpha mask to keep intensity between 0 and 1
        alpha = alpha.astype(float) / 255

        # cv2.imshow('alpha_crop', alpha)
        # cv2.waitKey(0)

        # multiply the profile with alpha matte
        img_profile_resized = cv2.multiply(alpha, img_profile_resized)

        # multiply the background with (1 - alpha)
        crop = cv2.multiply(np.subtract(1, alpha), crop)

        # add the masked profile and background
        res = cv2.add(img_profile_resized, crop)
        res = res.astype(np.uint8)

    return res


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
    random_seed = config["random_seed"]
    profile_preprocessing = config["profile_preprocessing"]
    background_preprocessing = config["background_preprocessing"]
    margin_range = config["margin_range"]
    start_idx = config["start_idx"]

    random.seed(random_seed)

    background_images = fu.find_images(directory=directory_backgrounds, extension='.jpg')

    # synthetically create data for each class
    class_dirs = [d for d in os.listdir(directory_profiles)]
    for class_dir in class_dirs:
        print("synthesizing class: {}".format(class_dir))

        class_dir_path = os.path.join(directory_profiles, class_dir)
        profile_images = fu.find_images(directory=class_dir_path, extension='.png')

        # create output directory for class
        fu.init_directory(directory=os.path.join(directory_output, class_dir))

        # loop until we have created num_images images
        img_idx = 0
        while img_idx < num_images:
            # randomly choose profile and background
            rand_img_profile = random.choice(profile_images)
            rand_img_background = random.choice(background_images)
            # print("rand_img_profile: {}".format(rand_img_profile))
            # print("rand_img_background: {}".format(rand_img_background))

            # read in profile and background images
            img_profile = cv2.imread(rand_img_profile, cv2.IMREAD_UNCHANGED) # we need the alpha channel for transparency
            img_background = cv2.imread(rand_img_background)

            # cv2.imshow('original profile', img_profile)
            # cv2.waitKey(0)
            #
            # cv2.imshow('original background', img_background)
            # cv2.waitKey(0)

            # apply preprocessing to profile and background images
            img_profile = preprocess_image(image_np=img_profile,
                                           preprocessing_dict=profile_preprocessing)

            # cv2.imshow('preprocessed background', img_background)
            # cv2.waitKey(0)

            # paste profile onto background
            res = combine_profile_and_mask(img_profile=img_profile,
                                           img_background=img_background,
                                           background_preprocessing_dict=background_preprocessing)
            # cv2.imshow('res', res)
            # cv2.waitKey(0)

            # save image to class directory
            if res is not None: # only write if we have a valid result
                print("{} / {}".format(img_idx + 1, num_images))
                out_path = os.path.join(directory_output, class_dir, "{}_syn.jpg".format(img_idx+start_idx))
                cv2.imwrite(out_path, res)
                img_idx += 1
            else:
                print("skipping incompatible img+bgrd combo")

    # copy config so we can recall what parameters were used to construct the dataset splits
    shutil.copy(yaml_path, os.path.join(directory_output, 'config.yml'))