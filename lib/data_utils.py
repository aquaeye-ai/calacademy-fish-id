"""
Data processing library
"""

# standard libs
import os
import cv2
import h5py
import ntpath
import random
import logging
import progressbar

import numpy as np

# personal libs
import lib.globals as globals
import lib.log_utils as log_utils
import lib.file_utils as file_utils
import image_splitting_utils as image_splitting_utils


logger = logging.getLogger(__name__)


@log_utils.timeit
def mse(image_A=None, image_B=None):
    """
    The 'Mean Squared Error' between the two images is the
    sum of the squared difference between the two images.
    Return the MSE.
    The lower the error, the more "similar" the two images are.
    NOTE: the two images must have the same dimension

    :param image_A: numpy array, data for image
    :param image_B: numpy array, data for image
    :return: float, error
    """
    assert(image_A.shape == image_B.shape)
    err = np.sum((image_A.astype("float") - image_B.astype("float")) ** 2)
    err /= float(image_A.shape[0] * image_A.shape[1])

    return err


@log_utils.timeit
def zero_center_data_globally(train_data=[], test_data=[]):
    """
    Zero centers the image data across all RGB channels.  Subtracts mean across all RGB channels in train_data from
    test_data.

    :param train_data:      list of training patches
    :param test_data:       list of test patches
    :return:                zero-centered test_data
    """

    mean = 0
    for patch in train_data:
        mean += np.mean(patch[0])

    mean /= len(train_data)
    mean = np.uint8(mean)

    new_test_data = []
    for idx, patch in enumerate(test_data):
        new_test_data.append((patch[0] - mean, patch[1], patch[2], patch[3], patch[4]))

    return new_test_data


# @log_utils.timeit
def randomly_scale_patches(patches=[], min=0.5, max=1.5):
    """
    Randomly scale patches of data between min and max.

    :param patches: patches to be scaled
    :param min:     minimum allowed scaling
    :param max:     maximum allowed scaling
    :return:        scaled patches
    """
    scaled_patches = []

    # bar = progressbar.ProgressBar()
    # for patch in bar(patches):
    for patch in patches:
        h_1, w_1, d_1 = patch.shape

        frame = np.zeros(patch.shape, dtype=np.uint8)

        scale_factor = random.uniform(min, max)

        scaled_patch = cv2.resize(patch, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

        h_2, w_2, d_2 = scaled_patch.shape

        # adjust size h,w by 1 pixel in case the original dimensions were odd integers (just an annoying "edge" case)
        if h_2 < h_1 and w_2 < w_1:
            x = 1 if (h_2 % 2 != 0) else 0
            y = 1 if (w_2 % 2 != 0) else 0
            frame[(h_1 / 2 - h_2 / 2):(h_1 / 2 + h_2 / 2 + x),
            (w_1 / 2 - w_2 / 2):(w_1 / 2 + w_2 / 2 + y)] = scaled_patch
        else:
            x = 1 if (h_1 % 2 != 0) else 0
            y = 1 if (w_1 % 2 != 0) else 0
            frame[:, :, :] = scaled_patch[(h_2 / 2 - h_1 / 2):(h_1 / 2 + h_2 / 2 + x),
                             (w_2 / 2 - w_1 / 2):(w_1 / 2 + w_2 / 2 + y), :]

        scaled_patches.append(frame)

    return scaled_patches


# @log_utils.timeit
def randomly_rotate_patches(patches=[]):
    """
    Randomly rotate patches between 0 and 360 degrees.

    :param patches: patches to be rotated
    :return:        rotated patches
    """
    rotated_patches = []
    rotation_angles = []

    # bar = progressbar.ProgressBar()
    # for patch in bar(patches):
    for patch in patches:
        h_1, w_1, d_1 = patch.shape
        (cX, cY) = (w_1 // 2, h_1 // 2)

        # rotated_patch = np.zeros(patch.shape, dtype=np.uint8)

        # randomly rotate patch (note: we don't care if edges/corners are clipped during rotation since
        # the final size needs to be fixed anyway)
        rotation_angle = random.uniform(0, 360)

        M = cv2.getRotationMatrix2D((cX, cY), rotation_angle, 1.0)

        rotated_patch = cv2.warpAffine(patch, M, (w_1, h_1))

        rotated_patches.append(rotated_patch)
        rotation_angles.append(rotation_angle)

    return rotated_patches, rotation_angles


@log_utils.timeit
def randomly_scale_rotate_patches(patches=[]):
    new_patches = []

    bar = progressbar.ProgressBar()
    for patch in bar(patches):
        h_1, w_1, d_1 = patch.shape
        (cX, cY) = (w_1 // 2, h_1 // 2)

        frame = np.zeros(patch.shape, dtype=np.uint8)

        scale_factor = random.uniform(0.5, 1.5)

        scaled_patch = cv2.resize(patch, None, fx=scale_factor, fy=scale_factor,
                                  interpolation=cv2.INTER_LINEAR)

        h_2, w_2, d_2 = scaled_patch.shape

        # adjust size h,w by 1 pixel in case the original dimensions were odd integers (just an annoying "edge" case)
        if h_2 < h_1 and w_2 < w_1:
            x = 1 if (h_2 % 2 != 0) else 0
            y = 1 if (w_2 % 2 != 0) else 0
            frame[(h_1 / 2 - h_2 / 2):(h_1 / 2 + h_2 / 2 + x),
            (w_1 / 2 - w_2 / 2):(w_1 / 2 + w_2 / 2 + y)] = scaled_patch
        else:
            x = 1 if (h_1 % 2 != 0) else 0
            y = 1 if (w_1 % 2 != 0) else 0
            frame[:, :, :] = scaled_patch[(h_2 / 2 - h_1 / 2):(h_1 / 2 + h_2 / 2 + x),
                             (w_2 / 2 - w_1 / 2):(w_1 / 2 + w_2 / 2 + y), :]

        rotation_angle = random.uniform(0, 360)

        M = cv2.getRotationMatrix2D((cX, cY), rotation_angle, 1.0)

        frame[:, :, :] = cv2.warpAffine(frame, M, (w_1, h_1))

        new_patches.append(frame)

    return new_patches


@log_utils.timeit
def zero_center_data_per_channel(train_data=[], test_data=[]):
    """
    Zero centers the image data per RGB channel.  Subtracts mean per RGB channel in train_data from each RGB channel in
    test_data respectively.

    :param train_data:      list of training patches
    :param test_data:       list of test patches
    :return:                zero-centered test_data
    """

    h, w = train_data[0][0].shape[0:2]

    # channel 0 mean
    mean_channel_0 = 0
    for patch in train_data:
        mean_channel_0 += np.sum(patch[0][:][:][0])

    mean_channel_0 /= (len(train_data) * h * w)
    mean_channel_0 = np.uint8(mean_channel_0)

    # channel 1 mean
    mean_channel_1 = 0
    for patch in train_data:
        mean_channel_1 += np.sum(patch[0][:][:][1])

    mean_channel_1 /= (len(train_data) * h * w)
    mean_channel_1 = np.uint8(mean_channel_1)

    # channel 2 mean
    mean_channel_2 = 0
    for patch in train_data:
        mean_channel_2 += np.sum(patch[0][:][:][2])

    mean_channel_2 /= (len(train_data) * h * w)
    mean_channel_2 = np.uint8(mean_channel_2)

    # subtract mean of each channel in train_data from each channel in test_data
    new_test_data = []
    for idx, patch in enumerate(test_data):
        new_patch = np.zeros(patch[0].shape, dtype=np.uint8)

        new_patch[:][:][0] = patch[0][:][:][0] - mean_channel_0
        new_patch[:][:][1] = patch[0][:][:][1] - mean_channel_1
        new_patch[:][:][2] = patch[0][:][:][2] - mean_channel_2

        new_test_data.append((new_patch, patch[1], patch[2], patch[3], patch[4]))

    return new_test_data


def partition_data(all_data=[], all_labels=[]):
    """
    Partition the data into training and testing splits, using 75%
    of data for training and the remaining 25% for testing
    """
    train_data = []
    test_data = []
    train_labels = []
    test_labels = []
    for i in range(len(all_data)):
        rn = random.random()
        if rn <=0.75:
            train_data.append(all_data[i])
            train_labels.append(all_labels[i])
        else:
            test_data.append(all_data[i])
            test_labels.append(all_labels[i])
    return (train_data, train_labels, test_data, test_labels)


def partition_data_two_labels(all_data=[], all_labels_1=[], all_labels_2=[]):
    """
    Partition the data into training and testing splits, using 75%
    of data for training and the remaining 25% for testing
    """
    train_data = []
    test_data = []
    train_labels_1 = []
    train_labels_2 = []
    test_labels_1 = []
    test_labels_2 = []
    for i in range(len(all_data)):
        rn = random.random()
        if rn <=0.75:
            train_data.append(all_data[i])
            train_labels_1.append(all_labels_1[i])
            train_labels_2.append(all_labels_2[i])
        else:
            test_data.append(all_data[i])
            test_labels_1.append(all_labels_1[i])
            test_labels_2.append(all_labels_2[i])
    return (train_data, train_labels_1, train_labels_2, test_data, test_labels_1, test_labels_2)


def hdf5_db_sequential_generator(dataset_type=None, hdf5_db=None):
    """
    Generator that iterates and returns all dimensions in hdf5_db database.

    Note: Will NOT infinitely loop over database.

    :param dataset_type:    DB_TYPES type, dataset type, must be found in DB_MAP
    :param hdf5_db:         str, path to hdf5 database to be used
    :return:                batch of: patch, edge labels, roof indicator labels, orientation codes
    """

    mapped_dataset_type = globals.DB_MAP[dataset_type]

    with h5py.File(hdf5_db, "r") as db:
        db_size = db["{}/patch".format(mapped_dataset_type)].len()

        for idx in xrange(db_size):
            ret = []

            for dim in globals.DB_DIM_TYPES:
                mapped_dataset_dimension = globals.DB_DIM_MAP[dim]

                # (patch, label, image_name, j, i, orientation_code, roof_indicator, roof_percentage)
                item = db["{}/{}".format(mapped_dataset_type, mapped_dataset_dimension)][idx]

                ret.append(item)

            yield tuple(ret)


def hdf5_batch_generator_simple(dataset_type=None, batch_size=None, hdf5_db=None):

    while 1:

        mapped_dataset_type = globals.DB_MAP[dataset_type]

        with h5py.File(hdf5_db, "r") as db:
            db_size = db["{}/patch".format(mapped_dataset_type)].len()

            # gather buffer sized chunks of data from hdf5 db as long as we have samples left in db
            num_batch = 0
            while num_batch < db_size:

                # edge case, leftover isn't integer multiple of desired buffer_size
                if num_batch < (db_size - batch_size):
                    current_batch_size = batch_size
                else:
                    current_batch_size = db_size - num_batch

                # (patch, label, image_name, j, i, orientation_code, roof_indicator, roof_percentage)
                patch_batch             = db["{}/patch".format(mapped_dataset_type)][num_batch:num_batch + current_batch_size]
                label_batch             = db["{}/label".format(mapped_dataset_type)][num_batch:num_batch + current_batch_size]
                image_name_batch        = db["{}/image_name".format(mapped_dataset_type)][num_batch:num_batch + current_batch_size]
                j_batch                 = db["{}/j".format(mapped_dataset_type)][num_batch:num_batch + current_batch_size]
                i_batch                 = db["{}/i".format(mapped_dataset_type)][num_batch:num_batch + current_batch_size]
                orientation_code_batch  = db["{}/orientation_code".format(mapped_dataset_type)][num_batch:num_batch + current_batch_size]
                roof_indicator_batch    = db["{}/roof_indicator".format(mapped_dataset_type)][num_batch:num_batch + current_batch_size]
                roof_percentage_batch   = db["{}/roof_percentage".format(mapped_dataset_type)][num_batch:num_batch + current_batch_size]

                num_batch += len(patch_batch)

                yield (patch_batch, label_batch, image_name_batch, j_batch, i_batch, orientation_code_batch,
                       roof_indicator_batch, roof_percentage_batch)


def hdf5_batch_generator(dataset_type=None, buffer_size=None, batch_size=None, hdf5_db=None):
    """
    Generator that returns batches of data from hdf5 file.
    Operates by consuming buffer sized chunks of data from file into memory, then yields batch sized chunks from
    in memory buffer.

    Note: Will NOT infinitely loop over database.

    :param dataset_type:    DB_TYPES type, dataset type, must be found in DB_MAP
    :param buffer_size:     int, amount of data to pull into memory, batches will then be yielded from this
                            in-memory buffer; Note: best to keep as multiple of batch_size
    :param batch_size:      int, size of batches to yield (should be kept close to 'chunk' size used creating
                            hdf5 database)
    :param hdf5_db:         str, path to hdf5 database to be used
    :return:                batch of: patchs, edge label, image_names, j coordinates, i coordinates,
                            orientation_codes, roof_indicators, roof_percentages
    """

    mapped_dataset_type = globals.DB_MAP[dataset_type]

    with h5py.File(hdf5_db, "r") as db:
        db_size = db["{}/patch".format(mapped_dataset_type)].len()

        # gather buffer sized chunks of data from hdf5 db as long as we have samples left in db
        num_buffer = 0 # number of elements buffered so far
        while num_buffer < db_size:

            # edge case, leftover isn't integer multiple of desired buffer_size
            if num_buffer < (db_size - buffer_size):
                current_buffer_size = buffer_size
            else:
                current_buffer_size = db_size - num_buffer

            # (patch, label, image_name, j, i, orientation_code, roof_indicator, roof_percentage)
            patch_buffer = db["{}/patch".format(mapped_dataset_type)][num_buffer:num_buffer + current_buffer_size]
            label_buffer = db["{}/label".format(mapped_dataset_type)][num_buffer:num_buffer + current_buffer_size]
            image_name_buffer = db["{}/image_name".format(mapped_dataset_type)][num_buffer:num_buffer + current_buffer_size]
            j_buffer = db["{}/j".format(mapped_dataset_type)][num_buffer:num_buffer + current_buffer_size]
            i_buffer = db["{}/i".format(mapped_dataset_type)][num_buffer:num_buffer + current_buffer_size]
            orientation_code_buffer = db["{}/orientation_code".format(mapped_dataset_type)][
                                      num_buffer:num_buffer + current_buffer_size]
            roof_indicator_buffer = db["{}/roof_indicator".format(mapped_dataset_type)][
                                    num_buffer:num_buffer + current_buffer_size]
            roof_percentage_buffer = db["{}/roof_percentage".format(mapped_dataset_type)][
                                     num_buffer:num_buffer + current_buffer_size]

            num_buffer += len(patch_buffer)

            # gather batch_size sized chunks of data from buffer
            rn_idx_array = random.sample(range(current_buffer_size), current_buffer_size)
            num_batch = 0 # number of elements batched so far for current buffer
            while num_batch < current_buffer_size:

                # edge case, leftover isn't integer multiple of desired batch_size
                if num_batch < (current_buffer_size - batch_size):
                    current_batch_size = batch_size
                else:
                    current_batch_size = current_buffer_size - num_batch

                patch_batch = []
                label_batch = []
                image_name_batch = []
                j_batch = []
                i_batch = []
                orientation_code_batch = []
                roof_indicator_batch = []
                roof_percentage_batch = []

                for i in xrange(current_batch_size):
                    rn_idx = rn_idx_array[-1:]

                    patch_batch.append(patch_buffer[rn_idx[0]])
                    label_batch.append(label_buffer[rn_idx[0]])
                    image_name_batch.append(image_name_buffer[rn_idx[0]])
                    j_batch.append(j_buffer[rn_idx[0]])
                    i_batch.append(i_buffer[rn_idx[0]])
                    orientation_code_batch.append(orientation_code_buffer[rn_idx[0]])
                    roof_indicator_batch.append(roof_indicator_buffer[rn_idx[0]])
                    roof_percentage_batch.append(roof_percentage_buffer[rn_idx[0]])

                    del rn_idx_array[-1:]  # we are sampling without replacement

                num_batch += current_batch_size

                yield (patch_batch, label_batch, image_name_batch, j_batch, i_batch, orientation_code_batch,
                       roof_indicator_batch, roof_percentage_batch)


def split_images_into_tiles(src_dir=None, max_size=500, margin=0, remove_original=False):
    """
    Splits directory of images into tiles, save those tiles in the original images' directory and if desired, remove
    the original images.

    :param src_dir:         str, path to images
    :param max_size:        int, tile size
    :param margin:          int, margin to add to each border of tile
    :param remove_original: bool, whether to remove the original images
    :return:                None
    """
    import lib.file_utils as file_utils # to avoid circular importing

    image_names = os.listdir(src_dir)
    image_names.sort()

    for image_name in image_names:
        image_path = os.path.join(src_dir, image_name)
        image = cv2.imread(image_path)

        tiles = image_splitting_utils.split_image_into_patches(image=image, max_size=max_size, margin=margin)

        for idx, tile in enumerate(tiles):
            file_utils.save_images([(image_path, tile)], prefix="", postfix="-tile-{}".format(idx), extension=".png")

        if remove_original:
            os.remove(image_path)


@log_utils.timeit
def map_k_channels_to_qtdt(arr=None):
    """
    For a MxNxK array, return the index of the max along each channel in depth vector (MxN such vectors) in the form
    of an MxN array.  Particularly useful for viz for roof+qtdt model
    :param arr: array, MxNxK
    :return: array, MxN
    """
    h, w, d = arr.shape
    res = np.zeros((h, w), np.uint8)

    def channel_max_index(a):
        i = a.argmax()
        a[0] = i # store in channel 0 for retrieval later
        return a

    arr = np.apply_along_axis(channel_max_index, axis=2, arr=arr)

    res[:,:] = arr[:, :, 0] # we stored the appropriate index in channel 0

    return res
