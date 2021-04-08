"""
Library for basic file helpers
"""

# standard libs
import os
import cv2
import errno
import ntpath
import shutil
import logging

import xml.etree.ElementTree as et

import numpy as np
import pandas as pd

# personal libs
import lib.scripts.globals as globals
import lib.scripts.log_utils as log_utils

logger = logging.getLogger(__name__)


def find_image_mask_json_tuples_for_tree_and_roof(images_dir=None, masks_roof_dir=None, masks_tree_dir=None, jsons_dir=None,
                                mask_roof_extension='ROOF.PNG', mask_tree_extension='TREE.PNG'):
    """
    Return list of tuples where each tuple contains the absolute paths for: image, mask and json.

    :param images_dir:      string, directory of image files
    :param masks_dir:       string, directory of mask files
    :param jsons_dir:       string, directory json files
    :param mask_extensions: list of str, extensions to match against when looking through masks_dir
    :return:            list of tuples -> (abs_image_path, abs_mask_path, abs_annotation_path)
    """
    image_files     = os.listdir(images_dir)
    mask_roof_files = os.listdir(masks_roof_dir)
    mask_tree_files = os.listdir(masks_tree_dir)
    json_files      = os.listdir(jsons_dir)

    image_files.sort()
    mask_roof_files.sort()
    mask_tree_files.sort()
    json_files.sort()

    tuples = []

    # for each image file, look for matching mask and json files

    for image_file in image_files:
        if image_file.upper().endswith('.PNG') and 'with-lines' not in image_file and 'mask' not in image_file:
            # get full image path
            image_path = os.path.join(images_dir, image_file)

            # get absolute image path
            image_path = os.path.abspath(image_path)

            # look for corresponding json
            ann = image_file[:-len('.PNG')] + '_annotation_'
            annlist = []
            for a in json_files:
                if a.startswith(ann) and a.upper().endswith('.JSON'):
                    annlist.append(a)

            # get full json path
            annotation_path = os.path.join(jsons_dir, annlist[-1])

            # get absolute json path
            annotation_path = os.path.abspath(annotation_path)

            # look for cooresponding roof mask
            mask_roof       = image_file[:-len('.PNG')]
            mask_roof_path  = ""
            for m in mask_roof_files:
                if m.startswith(mask_roof) and m.upper().endswith(tuple(mask_roof_extension)):
                    # get full mask path
                    mask_roof_path = m

            # get full mask path
            mask_roof_path = os.path.join(masks_roof_dir, mask_roof_path)

            # get absolute mask path
            mask_roof_path = os.path.abspath(mask_roof_path)

            # look for cooresponding tree mask
            mask_tree = image_file[:-len('.PNG')]
            mask_tree_path = ""
            for m in mask_tree_files:
                if m.startswith(mask_tree) and m.upper().endswith(tuple(mask_tree_extension)):
                    # get full mask path
                    mask_tree_path = m

            # get full mask path
            mask_tree_path = os.path.join(masks_tree_dir, mask_tree_path)

            # get absolute mask path
            mask_tree_path = os.path.abspath(mask_tree_path)

            tuples.append((image_path, mask_roof_path, mask_tree_path, annotation_path))

    return tuples


def find_image_mask_json_tuples(images_dir=None, masks_dir=None, jsons_dir=None, mask_extensions=['ROOF.PNG']):
    """
    Return list of tuples where each tuple contains the absolute paths for: image, mask and json.

    :param images_dir:      string, directory of image files
    :param masks_dir:       string, directory of mask files
    :param jsons_dir:       string, directory json files
    :param mask_extensions: list of str, extensions to match against when looking through masks_dir
    :return:            list of tuples -> (abs_image_path, abs_mask_path, abs_annotation_path)
    """
    image_files = os.listdir(images_dir)
    mask_files  = os.listdir(masks_dir)
    json_files  = os.listdir(jsons_dir)

    image_files.sort()
    mask_files.sort()
    json_files.sort()

    tuples = []

    # for each image file, look for matching mask and json files

    for image_file in image_files:
        if image_file.upper().endswith('.PNG') and 'with-lines' not in image_file and 'mask' not in image_file:
            # get full image path
            image_path = os.path.join(images_dir, image_file)

            # get absolute image path
            image_path = os.path.abspath(image_path)

            # look for corresponding json
            ann = image_file[:-len('.PNG')] + '_annotation_'
            annlist = []
            for a in json_files:
                if a.startswith(ann) and a.upper().endswith('.JSON'):
                    annlist.append(a)

            # get full json path
            annotation_path = os.path.join(jsons_dir, annlist[-1])

            # get absolute json path
            annotation_path = os.path.abspath(annotation_path)

            # look for cooresponding mask
            mask = image_file[:-len('.PNG')]
            mask_path = ""
            for m in mask_files:
                if m.startswith(mask) and m.upper().endswith(tuple(mask_extensions)):
                    # get full mask path
                    mask_path = m

            # get full mask path
            mask_path = os.path.join(masks_dir, mask_path)

            # get absolute mask path
            mask_path = os.path.abspath(mask_path)

            tuples.append((image_path, mask_path, annotation_path))

    return tuples


def find_image_json_pairs(directory=None, extension=".png"):
    """
    Look in the given directory for image files (<name>.png) and
    matching json files (<name>_annotation_<version>.json)
    """

    files = os.listdir(directory)
    files.sort()
    pairs = []

    # first look for image files
    # for each image file, look for matching json file

    for f in files:
        if f.upper().endswith(extension.upper()) and 'with-lines' not in f and 'mask' not in f:
            fname = os.path.join(directory, f)
            ann = f[:-len(extension.upper())] + '_annotation_'
            annlist = []
            for a in files:
                if a.startswith(ann) and a.upper().endswith('.JSON'):
                    annlist.append(a)

            if annlist:
                pairs.append((fname, os.path.join(directory, annlist[-1])))

    return pairs


def find_image_json_pairs_separate_dirs(img_dir=None, ann_dir=None, extension=".png"):
    """
    Look in the given directory for image files (<name>.png) and
    matching json files (<name>_annotation_<version>.json)
    """

    imgs = os.listdir(img_dir)
    anns = os.listdir(ann_dir)

    files = imgs + anns

    files.sort()

    pairs = []

    # first look for image files
    # for each image file, look for matching json file

    for f in files:
        if f.upper().endswith(extension.upper()):
            fname = os.path.join(img_dir, f)
            ann = f[:-len(extension.upper())]
            annlist = []
            for a in files:
                if a.startswith(ann) and a.upper().endswith('.JSON'):
                    annlist.append(a)

            if annlist:
                pairs.append((fname, os.path.join(ann_dir, annlist[-1])))

    return pairs


def detect_unclean_image_dir(directory=None):
    """
    Look in the given directory for non png/json image files and return true if any found.
    Useful since we only consider .png images and user error can easily introduce extraneous image formats.
    """

    files = os.listdir(directory)
    files.sort()

    # first look for image files
    # for each image file, look for matching json file

    for f in files:
        if not f.upper().endswith('.PNG') and not f.upper().endswith('.JSON'):
            return True

    return False


@log_utils.timeit
def zip_prediction_patches_and_train_masks(prediction_tuples=None, hdf5_db=None):
    """
    :param prediction_tuples:   list of tuples: (file-name, prediction-2D-array, original-image)
    :param hdf5_db:             hdf5_db used for training
    :return:                    new list of prediction tuples: (file-name, prediction-2D-array, original-image,
                                training-mask-2D-array)
    """
    from lib.scripts.data_utils import hdf5_db_sequential_generator # to avoid circular imports

    logger.info('zipping prediction tuples and training masks...')
    new_prediction_tuples = []
    for ptuple in prediction_tuples:
        image_name = ptuple[0]
        image = cv2.imread(image_name)
        h, w = image.shape[0:2]
        train_mask = np.zeros((h, w), dtype=np.uint8)

        # (patch, label, image_name, j, i, orientation_code, roof_indicator, roof_percentage)
        gen = hdf5_db_sequential_generator(dataset_type=globals.DB_TYPES.TRAIN, hdf5_db=hdf5_db)
        for point in gen:
            if point[2] == image_name:
                x = point[3]
                y = point[4]
                train_mask[x, y] = 1  # pixel was used for training
        new_prediction_tuples.append(ptuple + (train_mask,))

    return new_prediction_tuples


@log_utils.timeit
def zip_prediction_images_and_test_masks(prediction_tuples=None, test_masks_dir=None, image_size=None):
    """
    :param prediction_tuples:   list of tuples: (file-path, prediction-2D-array, original-image)
    :param test_masks_dir:      string, directory of test masks
    :param image_size:          int, optional, resize size for test masks
    :return:                    new list of prediction tuples: (file-name, prediction-2D-array, original-image,
                                test-mask-2D-array)
    """

    logger.info('zipping prediction tuples and test masks...')
    new_prediction_tuples = []
    test_mask_filenames = os.listdir(test_masks_dir)

    for ptuple in prediction_tuples:
        image_name = ptuple[0]

        for test_mask_filename in test_mask_filenames:
            base_image_name = ntpath.basename(image_name)
            base_image_name = os.path.splitext(base_image_name)[0]

            base_test_mask_filename = ntpath.basename(test_mask_filename)
            base_test_mask_filename = os.path.splitext(base_test_mask_filename)[0]

            if base_test_mask_filename in base_image_name:
                test_mask = cv2.imread(os.path.join(test_masks_dir, test_mask_filename))

                # resize mask
                # needed for tasks such as basic unet which resizes images+masks as a part of preprocessing
                if image_size is not None:
                    test_mask = cv2.resize(test_mask, (image_size, image_size))

                # reshape to (h, w) instead of (h, w, 1)
                test_mask = test_mask[:, :, 0]

                # rescale since we are dealing with original and did not save the preprocessed versions from training
                test_mask = test_mask / 255

                new_prediction_tuples.append(ptuple + (test_mask,))

    return new_prediction_tuples


def rename_annotation_files_to_match_originals(src_dir=None):
    """
    Rename each annotation image in src_dir by removing '_annotation...' from its name.
    May be necessary for tasks such as correctly pairing image+mask generators for keras.

    :param src_dir: string, directory of annotation images to rename
    :return:        None
    """
    files = os.listdir(src_dir)

    for file in files:
        base_name = file.split("_annotation")[0]
        base_name += '.png'

        old_file = os.path.join(src_dir, file)
        new_file = os.path.join(src_dir, base_name)

        # rename file
        os.rename(old_file, new_file)


def find_jsons(directory=None, extension=".json"):
    """
    Look in the given directory for json files

    :param directory: directory path
    :return: list of json file names
    """
    import warnings
    warnings.warn("deprecated", DeprecationWarning)

    files = os.listdir(directory)
    files.sort()
    jsons = []

    for f in files:
        if f.upper().endswith(extension.upper()):
            fname = os.path.join(directory, f)
            jsons.append(fname)

    return jsons


def find_files(directory=None, extension=".xml"):
    """
    Look in the given directory for files ending with the given extension.

    :param directory: directory path
    :return: list of file names
    """

    files = os.listdir(directory)
    files.sort()
    jsons = []

    for f in files:
        if f.upper().endswith(extension.upper()):
            fname = os.path.join(directory, f)
            jsons.append(fname)

    return jsons


def find_images(directory=None, extension=".png"):
    """
    Look in the given directory for image files
    Skip annotated, skip prediction images

    :param directory: directory path
    :return: list of image file names
    """

    files = os.listdir(directory)
    files.sort()
    images = []

    for f in files:
        if f.upper().endswith(extension.upper()) and 'with-lines' not in f and 'mask' not in f and '-pred' not in f:
            fname = os.path.join(directory, f)
            images.append(fname)

    return images


def show_images(images=None, delay=100):
    """
    Iterate through list of tuples: (image_file, image) and show each for delay
    """
    for pair in images:
        cv2.imshow('image', pair[1])
        cv2.waitKey(delay)


def save_images(images=None, prefix="", postfix="", extension=".png"):
    """
    Iterate through list of tuples: (image_path, image) and save each
    with corresponding prefix and postfix

    :param images:      list of tuples (image_path, image))
    :param prefix:      str, prefix to add to final image name
    :param postfix:     str, postfix to add to final image name
    :param extension:   str, file extension to use
    """

    for pair in images:
        name        = pair[0]
        path, ext   = os.path.splitext(name)
        basename    = os.path.basename(path)
        dir         = os.path.dirname(name)
        name        = "{}{}{}{}".format(prefix, basename, postfix, extension)
        out_path    = os.path.join(dir, name)
        cv2.imwrite(out_path, pair[1])


def find_boundary_images(directory=None):
    """
    Finds roof boundary images (generated separately in fish_from_boundaries.py) in given directory.

    :param directory:   str, directory to search in
    :return:            list, list of image file names
    """

    files = os.listdir(directory)
    files.sort()
    images = []

    for f in files:
        if f.upper().endswith('.ROOF.PNG'):
            fname = directory + f
            images.append(fname)

    return images


def remove_files_with_extension(directory=None, extension=None, use_logger=False):
    """
    roof_from_boundary.py generates both roof.png files as well as roof+image.png files.  This function removes the
    latter from the given directory
    :param directory:   str, directory to look
    :param extension:   str, extension to search to match against
    :param use_logger:  bool, whether to use print() or logger
    :return:            None
    """
    files = os.listdir(directory)
    files.sort()

    for f in files:
        if f.upper().endswith(extension.upper()): #case-insensitive
            fname = directory + f
            if use_logger:
                logger.info("removing: {}".format(fname))
            else:
                print("removing: {}".format(fname))
            os.remove(fname)


def symlink_images_to_directory(src_dir=None, dst_dir=None, use_logger=False):
    files = os.listdir(src_dir)
    files.sort()

    for f in files:
        if f.upper().endswith('.PNG') or f.upper().endswith('.JPG'):
            src_path = src_dir + f
            dst_path = dst_dir + f
            if use_logger:
                logger.info("symlinking: {} to {}".format(src_path, dst_path))
            else:
                print("symlinking: {} to {}".format(src_path, dst_path))
            os.symlink(src_path, dst_path)


def copy_images_to_directory(src_dir=None, dst_dir=None, use_logger=False):
    files = os.listdir(src_dir)
    files.sort()

    for f in files:
        if f.upper().endswith('.PNG') or f.upper().endswith('.JPG'):
            src_path = src_dir + f
            dst_path = dst_dir + f
            if use_logger:
                logger.info("symlinking: {} to {}".format(src_path, dst_path))
            else:
                print("symlinking: {} to {}".format(src_path, dst_path))
            shutil.copyfile(src_path, dst_path)


def init_directory(directory=None, use_logger=False):
    """
    Creates directory if doesn't already exist

    :param directory:   str, directory path
    :return:            str, directory path
    """

    try:
        os.makedirs(directory)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            if use_logger:
                logger.debug("os.makedirs failed on {}".format(directory))
            else:
                print("os.makedirs failed on {}".format(directory))
            raise

    return directory


def prune_outputs_directories():
    import lib.scripts.globals as globals
    import shutil

    keep_directories = []
    for keep_dir in os.listdir(globals.log_dir):
        keep_directories.append(keep_dir)

    print('directories to keep: {}'.format(keep_directories))

    parent_dirs = []
    parent_dirs.append(globals.data_visuals_dir)
    parent_dirs.append(globals.filter_visuals_dir)
    parent_dirs.append(globals.layer_visuals_dir)
    parent_dirs.append(globals.training_masks_dir)

    for parent_dir in parent_dirs:
        print('\noutputs directory to clean: {}'.format(parent_dir))

        for child_dir in os.listdir(parent_dir):
            if child_dir not in keep_directories:
                print('deleting: {}'.format(child_dir))

                shutil.rmtree(parent_dir + child_dir)


def renumber_image_bbox_slicer_output_files(source_directory=None, destination_directory=None, start_number=0, image_extension='.png'):
    """
    Renumbers annotation tiles' names sequentially starting from a given number.

    :param source_directory: str, path to directory holding original tiles
    :param destination_directory: str, path to directory to save renamed tiles
    :param start_number: int, starting value to begin sequentially renumbering from
    :param image_extension: str, image extension
    :return: None
    """
    image_paths = find_images(directory=source_directory, extension=image_extension)

    for idx, image_path in enumerate(image_paths):
        dir_path = os.path.dirname(image_path)
        basename = os.path.basename(image_path)
        src_ann_path = os.path.join(dir_path, basename[:-4]+".xml") # remove image extension from basename then add annotation extension
        dst_ann_path = os.path.join(destination_directory, "{}.xml".format(idx+start_number))
        dst_img_path = os.path.join(destination_directory, "{}{}".format(idx+start_number, image_extension))

        # copy files
        print("moving {} to {}".format(src_ann_path, dst_ann_path))
        print("moving {} to {}\n".format(image_path, dst_img_path))

        shutil.copyfile(src_ann_path, dst_ann_path)
        shutil.copyfile(image_path, dst_img_path)

        # adjust the filename and path tags in the xml file to match new destination
        et = xml.etree.ElementTree.parse(dst_ann_path)
        root = et.getroot()

        root.find("filename").text = "{}{}".format(idx+start_number, image_extension)
        root.find("path").text = dst_img_path
        et.write(dst_ann_path)


def remove_empty_annotations_and_images(directory=None, image_extension='.png', annotation_extension='.xml'):
    """
    Look through directory and remove empty annotation/image pairs.

    :param directory: str, directory path
    :param image_extension: str, e.g. '.png'
    :param annotation_extension: str, e.g. '.xml'
    :return: None
    """
    ann_paths = find_files(directory=directory, extension=annotation_extension)

    for idx, ann_path in enumerate(ann_paths):
        et = xml.etree.ElementTree.parse(ann_path)
        root = et.getroot()

        if root.find("object") == None:
            print("Found empty annotation/image pair")

            basename = os.path.basename(ann_path)
            basename = basename[:-4] # remove extension

            img_path = os.path.join(directory, basename+image_extension)

            print("Removing empty annotation: {}".format(ann_path))
            os.remove(ann_path)

            print("Removing empty image: {}".format(img_path))
            os.remove(img_path)


def string_replace_in_annotation_class(directory=None, old_str=None, new_str=None, annotation_extension='.png'):
    """
    Walks the xml tree and replaces any substring in the annotation's class label that matches old_str with new_str.

    :param directory: str, path
    :param old_str: str, string to replace
    :param new_str: str, string to replace old_str with
    :param annotation_extension: str, extension for annotation files
    :return: None
    """
    ann_paths = find_files(directory=directory, extension=annotation_extension)

    for idx, ann_path in enumerate(ann_paths):
        et = xml.etree.ElementTree.parse(ann_path)
        root = et.getroot()

        for elem in root.getiterator():
            if elem.tag == 'object':

                if old_str in elem._children[0].text:
                    print("Replacing '{}' in '{}' with '{}' for ann_path: {}".format(old_str, elem._children[0].text, new_str, ann_path))

                    elem._children[0].text = elem._children[0].text.replace(old_str, new_str)

        et.write(ann_path)


def collapse_classes_to_subset_in_csv(csv_path=None, keep_classes=None, default_class=None):
    """
    Collapse all classes in csv file (in TF training format such as that produced by lib/xml_to_csv_pbtxt.py) to
    default_class if they aren't present in the list provided by keep_classes.

    :param csv_path: str, path to csv file
    :param keep_classes: list of str, classes to keep
    :param default_class: str, class to collapse classes to that are not present in keep_classes
    :return: None
    """
    with open(csv_path) as csv_file:
        doc_df = pd.read_csv(csv_file)

        for row_idx, row in doc_df.iterrows():
            print("row: {}".format(row_idx))

            if doc_df['class'].values[row_idx] not in keep_classes:
                print('collapsing {} to {}'.format(doc_df['class'].values[row_idx], default_class))
                doc_df['class'][row_idx] = default_class

        doc_df.to_csv(csv_path)


def renumber_images(source_directory=None, destination_directory=None, start_number=0, image_extension='.png'):
    """
    Renumbers images' names sequentially starting from a given number.

    :param source_directory: str, path to directory holding original images
    :param destination_directory: str, path to directory to save renamed images
    :param start_number: int, starting value to begin sequentially renumbering from
    :param image_extension: str, image extension
    :return: None
    """
    init_directory(directory=destination_directory)

    image_paths = find_images(directory=source_directory, extension=image_extension)

    count = 0
    for idx, image_path in enumerate(image_paths):
        dst_img_path = os.path.join(destination_directory, "{}{}".format(idx+start_number, image_extension))

        # copy files
        print("moving {} to {}\n".format(image_path, dst_img_path))
        shutil.copyfile(image_path, dst_img_path)
        count += 1

    print("# images renumbered: {}".format(count))


def clean_image_extensions(source_directory=None, destination_directory=None):
    """
    Reduces image extensions to either .png or .jpg based on the presence of either in the extension.
    E.g. '<image_name>.jpg&w=600' -> '<image_name>.jpg' or '<image_name>.pngblaghgh12w' -> '<image_name>.png'

    Useful for cleaning the image extensions of scraped Bing/Google images (especially Bing) since some of their images
    tend to have strange file extensions.

    :param source_directory: str, path to directory holding original images
    :param destination_directory: str, path to directory to save renamed images
    :param start_number: int, starting value to begin sequentially renumbering from
    :param image_extension: str, image extension
    :return: None
    """
    from PIL import Image

    init_directory(directory=destination_directory)

    # grab all files in directory, since we don't know beforehand what their extensions will look like
    image_paths = os.listdir(source_directory)

    for idx, image_path in enumerate(image_paths):
        skip = False
        full_image_path = os.path.join(source_directory, image_path)

        name, dirty_extension = os.path.splitext(image_path)

        # choose appropriate file extension
        clean_extension = None
        image_type = Image.open(full_image_path).format
        if 'JPEG' == image_type:
            clean_extension = '.jpg'
        elif 'MPO' == image_type: # we cheat here and call it JPEG format since MPO is multiple jpg images combined into a stereo image
            clean_extension = '.jpg'
        elif 'PNG' == image_type:
            clean_extension = '.png'
        else:
            skip = True
            print('Unhandled image type: {} for image: {}'.format(image_type, full_image_path))

        # copy image to new path/extension
        if not skip:
            dst_img_path = os.path.join(destination_directory, "{}{}".format(name, clean_extension))

            # copy files
            # print("moving {} to {}\n".format(image_path, dst_img_path))
            shutil.copyfile(full_image_path, dst_img_path)

def bboxes_to_xml(image_path=None, image_np=None, detection_boxes=None, detection_classes=None, detection_scores=None,
                        category_index=None, min_score_threshold=0.7, dst_directory=None, label_blacklist=None):
    """
    Write bounding boxes and detection classes from output of object detection model for an image to annotation file
    readable by labelImg.

    :param image_path: str, path to image
    :param image_np: numpy array, image
    :param detection_boxes: list, list of detection box coordinates, coordinates are of form: [ymin, xmin, ymax, xmax]
                            where the values are percentages of image height/width
    :param detection_classes: list, list of detection classes that must be mapped using category_index
    :param detection_scores: list, list of scores for each bounding box
    :param category_index: dict, dictionary mapping integer values in detection_classes to readable strings
    :param min_score_threshold: float, threshold for bounding box to be written to annotation
    :param dst_directory: str, where to store image/annotation output
    :param label_blacklist: list, list of labels for which we ignore bounding boxes
    :return: None
    """
    import math

    h, w, d = image_np.shape
    image_name = ntpath.basename(image_path)
    ann_dst_path = os.path.join(dst_directory, image_name[:-len('.jpg')] + ".xml")

    # check to see if an existing annotation file exists
    # if one exists, use it otherwise create a new one
    root = None
    if os.path.isfile(image_path[:-len('.jpg')]+".xml"):
        tree = et.parse(image_path[:-len('.jpg')]+".xml")
        root = tree.getroot()
    else:
        # root
        root = et.Element("annotation")

        # boilerplate
        m1 = et.SubElement(root, "folder")
        m1.text = "temp"

        m2 = et.SubElement(root, "filename")
        m2.text = image_name

        m3 = et.SubElement(root, "path")
        m3.text = image_path

        m4 = et.SubElement(root, "source")
        m5 = et.SubElement(m4, "database")
        m5.text = "Unknown"

        m6 = et.SubElement(root, "size")
        m7 = et.SubElement(m6, "width")
        m7.text = "{}".format(w)
        m8 = et.SubElement(m6, "height")
        m8.text = "{}".format(h)
        m9 = et.SubElement(m6, "depth")
        m9.text = "{}".format(d)

        m10 = et.SubElement(root, "segmented")
        m10.text = "0"

    # bboxes
    for idx, bbox in enumerate(detection_boxes):
        bb_class = category_index[detection_classes[idx]]["name"]
        if detection_scores[idx] >= min_score_threshold and bb_class not in label_blacklist:
            bb_node = et.SubElement(root, "object")

            name = et.SubElement(bb_node, "name")
            name.text = bb_class

            pose = et.SubElement(bb_node, "pose")
            pose.text = "Unspecified"

            truncated = et.SubElement(bb_node, "truncated")
            truncated.text = "0"

            difficult = et.SubElement(bb_node, "difficult")
            difficult.text = "0"

            bndbox = et.SubElement(bb_node, "bndbox")
            xmin = et.SubElement(bndbox, "xmin")
            xmin.text = "{}".format(int(math.floor(bbox[1]*w)))
            ymin = et.SubElement(bndbox, "ymin")
            ymin.text = "{}".format(int(math.floor(bbox[0]*h)))
            xmax = et.SubElement(bndbox, "xmax")
            xmax.text = "{}".format(int(math.floor(bbox[3]*w)))
            ymax = et.SubElement(bndbox, "ymax")
            ymax.text = "{}".format(int(math.floor(bbox[2]*h)))

    tree = et.ElementTree(root)
    init_directory(directory=dst_directory)

    with open(ann_dst_path, "wb") as f:
        tree.write(f)

    img_dst_path = os.path.join(dst_directory, image_name)
    shutil.copyfile(src=image_path, dst=img_dst_path)