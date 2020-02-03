"""
Visualization library
"""


# standard libs
import os
import cv2
import json
import h5py
import ntpath
import logging
import matplotlib

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import pylab

# personal libs
import lib.globals as globals
import lib.log_utils as log_utils
import lib.file_utils as file_utils
import lib.data_utils as data_utils

from lib.annotate import generic_annotator


logger = logging.getLogger(__name__)



def sum_and_save(image_list, filename):
    """
    Create an 'average' image and save to a file.
    """

    logger.info("{} images were used to make {}".format(len(image_list), filename))
    if len(image_list) == 0:
        return

    sum = np.zeros((image_list[0].shape), dtype=float)
    for image in image_list:
        sum += image
    sum /= len(image_list)
    sum = sum.astype(np.uint8)

    sum = cv2.cvtColor(sum, cv2.COLOR_HSV2BGR)
    sum = cv2.resize(sum, None, fx=4, fy=4)
    cv2.imwrite(filename, sum)


def verify_line_labels(imagefile, jsonfile, dst_dir=globals.data_visuals_dir):
    """
    Load labels, create mask etc
    """

    logger.info('verifying: {}, {}'.format(imagefile, jsonfile))

    lookup = {'shingle-boundary': (0, 255, 255),
              'ridge': (0, 0, 255),
              'ridge-edge': (255, 0, 255),
              'outside-edge': (255, 0, 0),
              'inside-edge': (0, 255, 0),
              'occluding-edge': (255, 255, 255),
              'valley': (100, 0, 0),
              'hip': (0, 100, 100)
              }

    # load image
    image = cv2.imread(imagefile)
    h, w = image.shape[0:2]
    print(w, 'x', h)

    # load annotations
    annotation_data = {}
    with open(jsonfile) as data_file:
        annotation_data = json.load(data_file)

    # draw lines
    for idx, line in enumerate(annotation_data["lines"]):
        if line['label'] not in lookup:
            logger.info('Need to put a color in lookup fr label: {}'.format(line['label']))
        else:
            color = lookup[line['label']]
            x1 = line["start"][0]
            y1 = line["start"][1]
            x2 = line["end"][0]
            y2 = line["end"][1]
            width = line["thickness"]
            cv2.line(image, (x1, y1), (x2, y2), color, width)

    # save image
    base_name = ntpath.basename(imagefile)
    base_name = os.path.splitext(base_name)[0]
    outputfile = os.path.join(dst_dir, base_name + '-with-lines.png')
    cv2.imwrite(outputfile, image)


def prediction_errors_to_viz(error_tuples=[], postfix='-prediction-errors'):
    '''
    :param error_tuples: list of tuples containing errors from all predictions over all images trained on.
                            Tuple = (patch, label, image_name, j, i) where j,i are patch location in original image
    :param postfix:      string to append to saved image filename
    :return:             list of tuples: (image_file, image) where image is original image in channel 0 (B/W) and prediction errors
                                in channel 1 (255)
    '''
    # Form list of all images that errors stem from
    image_files = []
    for error in error_tuples:
        if error[2] not in image_files:
            image_files.append(error[2])

    viz_images = []
    for image_file in image_files:
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        h, w = image.shape[0:2]

        viz = np.zeros((h, w, 3), dtype=np.uint8)
        viz[:, :, 0] = image[:, :]  # original image in blue

        for error in error_tuples:
            if error[2] == image_file:
                x = error[3]
                y = error[4]
                viz[x, y, 2] = 255 # prediction error in red

        base_name = ntpath.basename(image_file)
        base_name = os.path.splitext(base_name)[0]
        image_path = globals.data_visuals_dir + '/' + base_name

        viz_images.append((image_path, viz))

        file_utils.save_images([(image_path, viz)], postfix=postfix)

    return viz_images


def prediction_errors_to_viz_hdf5(hdf5_db=None, dataset_type=None, postfix='-prediction-errors'):
    '''
    Given hdf5 database of error data (formatted to spec in lib/globals.py), returns set of images visualizing that data.

    :param hdf5_db:     str, database of tuples containing errors from all predictions over all images trained on
                        (assumed formatted/structured as noted in lib/globals.py)
    :param dataset_type DB_TYPES type, dataset type, must be found in DB_MAP
    :param postfix:     str, string to append to saved image filename
    :return:            list of tuples: (image_file, image) where image is original image in channel 0 (B/W) and
                        prediction errors in channel 1 (255)
    '''

    viz_images = []

    with h5py.File(hdf5_db, "a") as db:


        # form list of all images that errors stem from
        errors_generator = data_utils.hdf5_db_sequential_generator(dataset_type=dataset_type,
                                                                   hdf5_db=hdf5_db)

        image_files = []
        for idx, error in enumerate(errors_generator):
            # safety checks against empty file names
            assert error[globals.DB_DIM_TYPES.IMAGE_NAME.value][0] != ''
            assert error[globals.DB_DIM_TYPES.IMAGE_NAME.value][0] != ""
            assert error[globals.DB_DIM_TYPES.IMAGE_NAME.value][0] != None

            if error[globals.DB_DIM_TYPES.IMAGE_NAME.value][0] not in image_files: # data dimension for label stored as 1d ndarray hdf5 so have to access 0th element
                image_files.append(error[globals.DB_DIM_TYPES.IMAGE_NAME.value][0])


        # gather/display all errors for each image
        # have to create generator again since it was consumed creating image_files
        errors_generator = data_utils.hdf5_db_sequential_generator(dataset_type=dataset_type,
                                                                   hdf5_db=hdf5_db)

        for image_file in image_files:
            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            h, w = image.shape[0:2]

            viz = np.zeros((h, w, 3), dtype=np.uint8)
            viz[:, :, 0] = image[:, :]  # original image in blue

            for error in errors_generator:
                if error[globals.DB_DIM_TYPES.IMAGE_NAME.value][0] == image_file:
                    x = error[globals.DB_DIM_TYPES.J.value][0]
                    y = error[globals.DB_DIM_TYPES.I.value][0]
                    viz[x, y, 2] = 255 # prediction error in red

            base_name = ntpath.basename(image_file)
            base_name = os.path.splitext(base_name)[0]
            image_path = globals.data_visuals_dir + '/' + base_name

            viz_images.append((image_path, viz))

            file_utils.save_images([(image_path, viz)], postfix=postfix)

    return viz_images


def layer_filters_to_viz(layer=None):
    """
    Visualizes the weights of all filters of a given convolutional layer assuming the kernal size used was (3, 3, 3).
    Each neuron/kernel in a filter is presumed to be a (3, 3, 3) volume so we visualize each depth-wise slice of this neuron
    as three, 3x3 images per row in the final output image.

    :param layer:   keras layer handle
    :return:        plot handle
    """
    filters = layer.get_weights()[0]
    fig = plt.figure(figsize=(15, 160))
    num_filters = len(filters[0, 0, 0, :])
    for i in range(num_filters):
        f = filters[:, :, :, i]
        h, w, d = f.shape

        # gather each depth-wise slice of kernel
        chan_max_1 = np.amax(f[:, :, 0])
        chan_max_2 = np.amax(f[:, :, 1])
        chan_max_3 = np.amax(f[:, :, 2])

        # scale each depth-wise slice of the given kernel by that slice's maximum value
        viz = np.zeros((h, w, 3), dtype=np.uint8)
        viz[:, :, 0] = 255 * f[:, :, 0] / chan_max_1
        viz[:, :, 1] = 255 * f[:, :, 1] / chan_max_2
        viz[:, :, 2] = 255 * f[:, :, 2] / chan_max_3

        # plot the three depth-wise slices of the kernel across row in final figure
        for j in range(0, 3):
            ax = fig.add_subplot(num_filters, 3, i * 3 + j + 1)
            ax.matshow(viz[:, :, j], cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
    # plt.tight_layout()

    filename = "{}/filters".format(globals.filter_visuals_dir)
    pylab.savefig(filename, bbox_inches='tight')
    return plt


def layer_filters_to_viz_bw(layer=None):
    """
    Visualizes the weights of all filters of a given convolutional layer assuming the kernal size used was (3, 3, 1),
    i.e. black/white.  Each neuron/kernel in a filter is presumed to be a (3, 3, 1) volume so we visualize each
    depth-wise slice of this neuron as three, 3x3 images per row in the final output image.

    :param layer:   keras layer handle
    :return:        plot handle
    """
    filters = layer.get_weights()[0]
    fig = plt.figure(figsize=(15, 160))
    num_filters = len(filters[0, 0, 0, :])
    for i in range(num_filters):
        f = filters[:, :, :, i]
        h, w, d = f.shape

        # gather each depth-wise slice of kernel
        chan_max_1 = np.amax(f[:, :, 0])

        # scale each depth-wise slice of the given kernel by that slice's maximum value
        viz = np.zeros((h, w, 3), dtype=np.uint8)
        viz[:, :, 0] = 255 * f[:, :, 0] / chan_max_1

        # plot the three depth-wise slices of the kernel across row in final figure
        for j in range(0, 3):
            ax = fig.add_subplot(num_filters, 3, i * 3 + j + 1)
            ax.matshow(viz[:, :, j], cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
    # plt.tight_layout()

    filename = "{}/filters".format(globals.filter_visuals_dir)
    pylab.savefig(filename, bbox_inches='tight')
    return plt


@log_utils.timeit
def convert_predictions_to_viz(prediction_tuples=None, postfix="-edge-predictions"):
    """
    The tuples are (file-name, prediction-2D-array, original-image, mask)
    Convert to an RGB image where the prediction values are in one channel (0) [B for Opencv]
    The original image, converted to B/W is in another channel (1) [G]
    The pixels used in mask are labeled in the last channel(2) [R for OpenCV]
    One could also use the alpha-channel to mark the mask samples.
    """
    viz_images = []

    for ptuple in prediction_tuples:
        filename, prediction, original, mask = ptuple
        assert prediction.shape[0:1] == original.shape[0:1] # let number of channels differ since prediction could have
                                                            # as few as 1
        h, w, d, = original.shape
        assert d == 3

        viz = np.zeros((h, w, 3), dtype=np.uint8)
        viz[:, :, 0] = np.round(255 * prediction[:, :]) # prediction in channel 1
        viz[:, :, 1] = original[:, :, 2]  # original (grayscale) in channel 2
        viz[:, :, 2] = 255 * mask[:, :]  # mask (black/white) in channel 3

        base_name   = ntpath.basename(filename)
        base_name   = os.path.splitext(base_name)[0]
        image_path  = os.path.join(globals.data_visuals_dir, base_name)

        viz_images.append((image_path, viz))
        file_utils.save_images([(image_path, viz)], postfix=postfix)

    return viz_images


@log_utils.timeit
def convert_qtdt_predictions_to_viz(prediction_tuples=None, postfix="-qtdt-prediction", k_channels=False):
    """
    Right now just visualizes the prediction but not the original or mask as incorrectly stated below.
    The tuples are (file-name, prediction-2D-array, original-image, mask)
    Convert to an RGB image where the prediction values are in one channel (0) [B for Opencv]
    The original image, converted to B/W is in another channel (1) [G]
    The pixels used in mask are labeled in the last channel(2) [R for OpenCV]
    One could also use the alpha-channel to mark the mask samples.
    """
    viz_images = []

    for ptuple in prediction_tuples:
        filename, prediction, original, mask = ptuple
        assert prediction.shape[0:1] == original.shape[0:1] # let number of channels differ since prediction could have
                                                            # as few as 1
        # h, w, d, = original.shape
        # assert d == 3
        #
        # viz = np.zeros((h, w, 3), dtype=np.uint8)
        # viz[:, :, 0] = np.round(255 * prediction[:, :]) # prediction in channel 1
        # viz[:, :, 1] = original[:, :, 2]  # original (grayscale) in channel 2
        # viz[:, :, 2] = 255 * mask[:, :]  # mask (black/white) in channel 3

        # If dealing with model output instead of qtdt.py output, our prediction is MxNxK where K = number of qtdt bins.
        # (i, j, k) represent the probability of distance bin k at pixel (i,j).  So we look through prediction array,
        # find the indices in the depth dimension corresponding to the max in that dimension and collapse the MxNxK
        # array with values between [0, 1] into an MxN array with values in range [0, k-1] corresponding to pixel
        # distances for the qtdt for easier visualization.
        if k_channels:
            best_prediction = data_utils.map_k_channels_to_qtdt(prediction)
        else:
            best_prediction = prediction

        maximum = max(map(max, best_prediction)) # prediction is distance transform
        scaled = best_prediction / float(maximum) # avoid integer division
        quantized_distance_transform_heatmap = cv2.applyColorMap(np.uint8(255 * scaled), cv2.COLORMAP_JET)

        base_name   = ntpath.basename(filename)
        base_name   = os.path.splitext(base_name)[0]
        image_path  = os.path.join(globals.data_visuals_dir, base_name)

        viz_images.append((image_path, quantized_distance_transform_heatmap))
        file_utils.save_images([(image_path, quantized_distance_transform_heatmap)], postfix=postfix)

    return viz_images


@log_utils.timeit
def convert_preds_to_images(prediction_tuples=None, threshold=0.75, prediction_class=None):
    """
    Converts list of tuples: (image_file, 2D array) where 2D array corresponds to pixel probabilities
    to list of black/white images (for edge maps).

    :param preds:               list of tuples -> [(image_file, 2D prediction array)], prediction values in range [0-1]
    :param threshold:           (float) threshold applied to prediction values, if prediction > threshold then keep else throw
                                away
    :param prediction_class:    str, class under consideration, e.g. for roof_segmentation basic_unet it is "roof"
    :return:                    list of tuples -> [(filename, 2D thresholded prediction array converted to image)]
    """

    edge_maps = []

    for ptuple in prediction_tuples:
        pred = ptuple[1]

        pre_thresh_histo = np.histogram(pred, 20)
        logger.info("histogram of prediction values (pre-threshold): {}".format(pre_thresh_histo[0]))
        logger.info("histogram buckets: {}".format(pre_thresh_histo[1]))

        # convert prediction array (0-1) to pixel values (0-255)
        pre_thresh_img = np.array(pred * 255.0, dtype=np.uint8)  # threshold function expects type of either 8 bit ints or 32 bit floats

        base_name = ntpath.basename(ptuple[0])
        base_name = os.path.splitext(base_name)[0]
        image_path = os.path.join(globals.data_visuals_dir, base_name)

        postfix = "-pred-pre-thresh-{}".format(threshold)
        if prediction_class:
            postfix = "-{}-pred-pre-thresh-{}".format(prediction_class, threshold)
        file_utils.save_images([(image_path, pre_thresh_img)], postfix=postfix)

        # threshold image
        post_thresh_pred = (pred > threshold) * pred

        # convert prediction array (0-1) to pixel values (0-255)
        post_thresh_img = np.array(post_thresh_pred * 255.0, dtype=np.uint8)

        base_name = ntpath.basename(ptuple[0])
        base_name = os.path.splitext(base_name)[0]
        image_path = os.path.join(globals.data_visuals_dir, base_name)

        postfix = "-pred-post-thresh-{}".format(threshold)
        if prediction_class:
            postfix = "-{}-pred-post-thresh-{}".format(prediction_class, threshold)
        file_utils.save_images([(image_path, post_thresh_img)], postfix=postfix)

        post_thresh_histo = np.histogram(post_thresh_pred, 20)
        logger.info("histogram of prediction values (post-threshold): {}".format(post_thresh_histo[0]))
        logger.info("histogram buckets: {}".format(post_thresh_histo[1]))

        edge_maps.append((ptuple[0], post_thresh_img))

    return edge_maps


def crop_boundary_masks_and_images_to_minimum_bounding_box(file_tuples=None, line_labels=None):
    """
    Find the 4 coordinates to describe the smallest possible box that encapsulates all of the annotation lines.
    Crops images and corresponding masks closest to these 4 coordinates.

    :param file_tuples: list of tuples -> (image_name, image, mask, json)
    :param line_labels: list of str, strings are line labels to consider when producing bounding box from labels in
                        annotation
    :return:            None
    """

    logger.info("cropping original images and training masks...")

    for ftuple in file_tuples:
        image_path      = ftuple[0]
        mask_path       = ftuple[1]
        annotation_path = ftuple[2]

        image           = cv2.imread(image_path)
        mask            = cv2.imread(mask_path)

        json_data = {}
        with open(annotation_path) as json_file:
            json_data = json.load(json_file)

        h, w            = image.shape[0:2]
        # new_mask    = np.zeros((h, w), dtype=np.uint8)

        line_x_ranges = []
        line_y_ranges = []
        for index, line in enumerate(json_data["lines"]):
            if line['label'] in line_labels:
                x1 = line["start"][0]
                y1 = line["start"][1]
                x2 = line["end"][0]
                y2 = line["end"][1]

                if x1 != x2:  # ignore points marked by user
                    line_x_ranges.append([x1, x2])
                if y1 != y2:
                    line_y_ranges.append([y1, y2])

        # Find the 4 coordinates to describe the smallest possible box that encapsulates all of the annotation lines

        x_max, y_max = 0, 0
        x_min = w
        y_min = h

        # find x_max, x_min
        if len(line_x_ranges) > 0:
            for index, range_x in enumerate(line_x_ranges):
                current_max = max(range_x)
                current_min = min(range_x)
                if current_max > x_max:
                    x_max = current_max
                if current_min < x_min:
                    x_min = current_min
        else:  # there are no annotations so bounding box should be entire image
            x_max = w
            x_min = 0

        # find y_max, y_min
        if len(line_y_ranges) > 0:
            for index, range_y in enumerate(line_y_ranges):
                current_max = max(range_y)
                current_min = min(range_y)
                if current_max > y_max:
                    y_max = current_max
                if current_min < y_min:
                    y_min = current_min
        else:  # there are no annotations so bounding box should be entire image
            y_max = h
            y_min = 0

        # ensure that bounding box is within confines of image (was an edge case that I ran into)
        if x_min < 0:
            x_min = 0
        if y_min < 0:
            y_min = 0
        if x_max > w:
            x_max = w
        if y_max > h:
            y_max = h

        # widen bounding box by some arbitrary number of pixels to give lines buffer room (if possible).
        buffer = 50
        if y_min - buffer >= 0:
            y_min -= buffer
        if x_min - buffer >= 0:
            x_min -= buffer
        if x_max + buffer <= w:
            x_max += buffer
        if y_max + buffer <= h:
            y_max += buffer

        # perform cropping
        image   = image[y_min:y_max+1, x_min:x_max+1]
        mask    = mask[y_min:y_max+1, x_min:x_max+1]

        # overwrite the old image and mask
        cv2.imwrite(image_path, image)
        cv2.imwrite(mask_path, mask)


def pos_mask_from_json(image_path=None, json_path=None, save_to_dir=None):
    """
    For one image/json pair, produces and saves positive mask from annotations found in json.

    :param image:  image, image path to image to produce masks from
    :param json:   json file that holds annotation for mask
    :return:       2D array 0/1, pos mask
    """

    logger.info("constructing mask...")
    with open(json_path) as json_file:
        image       = cv2.imread(image_path)
        json_data   = json.load(json_file)
        h, w        = image.shape[0:2]
        mask        = np.zeros((h, w), dtype=np.uint8)

        line_x_ranges = []
        line_y_ranges = []
        for index, line in enumerate(json_data["lines"]):
            if line['label'] in generic_annotator:
                x1 = line["start"][0]
                y1 = line["start"][1]
                x2 = line["end"][0]
                y2 = line["end"][1]
                width = line["thickness"]
                cv2.line(mask, (x1, y1), (x2, y2), 255, width)

                if x1 != x2:  # ignore points marked by user
                    line_x_ranges.append([x1, x2])
                if y1 != y2:
                    line_y_ranges.append([y1, y2])

        # cv2.imshow('image', mask)
        # cv2.waitKey(1000)

        base_name = ntpath.basename(image_path)
        base_name = os.path.splitext(base_name)[0]

        logger.info("saving mask for image: {} to {}".format(base_name, save_to_dir))

        outputfile = "{}/{}.png".format(save_to_dir, base_name)
        cv2.imwrite(outputfile, mask)

        return mask