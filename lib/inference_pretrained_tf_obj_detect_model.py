import os
import yaml
import time
import logging
import tarfile

import cv2 as cv2
import numpy as np
import file_utils as fu
import tensorflow as tf
import six.moves.urllib as urllib
import lib.log_utils as log_utils

from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util

# configure logging
log_utils.LOG_DIR = "/home/nightrider/calacademy-fish-id/outputs"
log_utils.init_logging(file_name="inference_pretrained_tf_obj_detect_model_log.txt")

logger = logging.getLogger(__name__)

# # initialize logging
# log_utils.init_logging()
#
# # What model to download.
# MODEL_NAME = 'faster_rcnn_resnet101_coco_2018_01_28'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
#
# # Path to frozen detection graph. This is the actual model that is used for the object detection.
# PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
#
# # For the sake of simplicity we will use only 1 image:
# # image1.jpg
# # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
# PATH_TO_TEST_IMAGES_DIR = "/home/nightrider/calacademy-fish-id/datasets/pcr/stills/full/test"
# TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 2)] #TODO: use lib/file_utils.py
#
# # Size, in pixels of input image
# IMAGE_H = IMAGE_W = [600, 1024]
#
# # List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join('/home/nightrider/tensorflow/models/research/object_detection', 'data', 'mscoco_complete_label_map.pbtxt')
# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#     file_name = os.path.basename(file.name)
#     if 'frozen_inference_graph.pb' in file_name:
#         tar_file.extract(file, os.getcwd())
#
# detection_graph = tf.Graph()
# with detection_graph.as_default():
#     od_graph_def = tf.GraphDef()
#     with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
#         serialized_graph = fid.read()
#         od_graph_def.ParseFromString(serialized_graph)
#         tf.import_graph_def(od_graph_def, name='')
#
# # List of the strings that is used to add correct label for each box.
# category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

@log_utils.timeit
def run_inference_for_multiple_images(images, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            # TODO: add this back in
            # if 'detection_masks' in tensor_dict:
            #     # The following processing is only for single image
            #     detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            #     detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            #     # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            #     real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            #     detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            #     detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            #     detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            #         detection_masks,
            #         detection_boxes,
            #         image.shape[1],
            #         image.shape[2])
            #     detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            #     # Follow the convention by adding back the batch dimension
            #     tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            t1 = time.time()
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: images})
            t2 = time.time()
            logger.info("inference time: {}".format(t2-t1))

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'][:] = map(int, output_dict['num_detections'][:])
            output_dict['detection_classes'][:] = output_dict['detection_classes'][:].astype(np.int64)
            output_dict['detection_boxes'][:] = output_dict['detection_boxes'][:]
            output_dict['detection_scores'][:] = output_dict['detection_scores'][:]
            # TODO: add this back in
            # if 'detection_masks' in output_dict:
            #     output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

@log_utils.timeit
def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks,
                    detection_boxes,
                    image.shape[1],
                    image.shape[2])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.int64)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

def predict_images_tiled():
    for im_idx, image_path in enumerate(TEST_IMAGE_PATHS):
        print("image: {}".format(image_path))

        for k in range(0, len(IMAGE_H)):
            print("image resolution: {}".format(IMAGE_H[k]))

            image_np = cv2.imread(image_path)
            detection_scores = []
            detection_classes = []
            detection_boxes = []

            # Pad image dimensions to nearest multiple of 600 (for faster_rcnn_resent101) so that we can operate on crops
            h_mult = np.ceil(image_np.shape[0] / float(IMAGE_H[k]))
            w_mult = np.ceil(image_np.shape[1] / float(IMAGE_W[k]))
            h_new = h_mult * IMAGE_H[k]
            w_new = w_mult * IMAGE_W[k]
            h_pad_top = 0
            h_pad_bottom = h_new - image_np.shape[0]
            w_pad_right = w_new - image_np.shape[1]
            w_pad_left = 0
            image_pad_np = cv2.copyMakeBorder(image_np, int(h_pad_top), int(h_pad_bottom), int(w_pad_left), int(w_pad_right), borderType=cv2.BORDER_CONSTANT, value=0)

            # cv2.imshow('image_pad_np', image_pad_np)
            # cv2.waitKey()

            print("h_mult={}".format(h_mult))
            print("w_mult={}".format(w_mult))

            # Perform inference on tiles of image for better accuracy
            for i in range(0, int(h_mult)):
                for j in range(0, int(w_mult)):
                    tile_np = image_pad_np[i*IMAGE_H[k]:(i+1)*IMAGE_H[k], j*IMAGE_W[k]:(j+1)*IMAGE_W[k], :]

                    print("i={}, j={}".format(i, j))
                    # cv2.imshow('tile-i={}-j={}'.format(i, j), tile_np)
                    # cv2.waitKey()

                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(tile_np, axis=0)

                    # Actual detection.
                    output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)

                    # adjust the bounding box coordinates due to the tiling
                    # for box in non_zero_detection_boxes:
                    for idx, box in enumerate(output_dict['detection_boxes']):
                        h_offset = i*IMAGE_H[k]
                        w_offset = j*IMAGE_W[k]
                        ymin, xmin, ymax, xmax = box
                        tile_h, tile_w, tile_d = tile_np.shape[:]

                        # the box coordinates were normalized so we must get the pixel value of each
                        ymin *= tile_h
                        ymax *= tile_h
                        xmin *= tile_w
                        xmax *= tile_w

                        # adjust the box coordinates to be relative to the original image instead of tile
                        ymin += h_offset
                        ymax += h_offset
                        xmin += w_offset
                        xmax += w_offset

                        # update boxes list
                        output_dict['detection_boxes'][idx][:] = [ymin, xmin, ymax, xmax]

                    detection_classes.extend(output_dict['detection_classes'])
                    detection_scores.extend(output_dict['detection_scores'])
                    detection_boxes.extend(output_dict['detection_boxes'])

        # Visualization of the results of a detection.
        # This function groups boxes that correspond to the same location: https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py
        vis_util.visualize_boxes_and_labels_on_image_array(
            # tile_np,
            image_np,
            np.asarray(detection_boxes, dtype=np.float32),
            np.asarray(detection_classes, dtype=np.int64), # np arrays are double by nature, but this function requires ints for its classes
            np.asarray(detection_scores, dtype=np.float32),
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=False,  # we've adjusted the tiles' box coordinates
            line_thickness=8,
            min_score_thresh=0.1,
            max_boxes_to_draw=None) # None will force the function to look at all boxes in list which is what we want since our list of boxes isn't ordered in any way

        # save the original image with boxes
        basename = os.path.basename(image_path)[:-4] # get basename and remove extension of .png or .jpg
        out_image_np_path = "/home/nightrider/calacademy-fish-id/outputs/{}".format(basename)
        print("tile_np_path={}".format(out_image_np_path))
        fu.save_images(images=[(out_image_np_path, image_np)])

        ## save the detection classes and scores to text file
        # First we grab only non-zero probability detection outputs
        # Note: we don't remove duplicate boxes as this could affect our evaluation metrics
        non_zero_outputs = np.asarray(detection_scores, dtype=np.float32) > 0
        non_zero_detection_classes = np.asarray(detection_classes, dtype=np.int64)[non_zero_outputs] # indexing must be done on np array and not list
        non_zero_detection_scores = np.asarray(detection_scores, dtype=np.float32)[non_zero_outputs] # indexing must be done on np array and not list
        non_zero_detection_boxes = np.asarray(detection_boxes, dtype=np.float32)[non_zero_outputs] # indexing must be done on np array and not list

        out_image_np_text_path = "/home/nightrider/calacademy-fish-id/outputs/{}.txt".format(basename)
        out_image_np_text = open(out_image_np_text_path, "a+")
        for pr_tuple in zip(non_zero_detection_classes, non_zero_detection_scores, non_zero_detection_boxes):
            pr_class = category_index[pr_tuple[0]]["name"]
            out_image_np_text.write("{} {} {}\n".format(pr_class, pr_tuple[1], " ".join(map(str, pr_tuple[2]))))
        out_image_np_text.close()

@log_utils.timeit
def predict_images_batched(test_image_paths=None, input_image_sizes=None, category_index=None):
    for im_idx, image_path in enumerate(test_image_paths):
        logger.info("image: {}".format(image_path))

        for k in range(0, len(input_image_sizes)):
            image_size = input_image_sizes[k]
            logger.info("image size: {}x{}".format(image_size, image_size))

            tiles_np = []
            tile_ins = []

            image_np = cv2.imread(image_path)
            detection_scores = []
            detection_classes = []
            detection_boxes = []

            # Pad image dimensions to nearest multiple of 600 (for faster_rcnn_resent101) so that we can operate on crops
            h_mult = np.ceil(image_np.shape[0] / float(image_size))
            w_mult = np.ceil(image_np.shape[1] / float(image_size))
            h_new = h_mult * image_size
            w_new = w_mult * image_size
            h_pad_top = 0
            h_pad_bottom = h_new - image_np.shape[0]
            w_pad_right = w_new - image_np.shape[1]
            w_pad_left = 0
            image_pad_np = cv2.copyMakeBorder(image_np, int(h_pad_top), int(h_pad_bottom), int(w_pad_left), int(w_pad_right), borderType=cv2.BORDER_CONSTANT, value=0)

            # cv2.imshow('image_pad_np', image_pad_np)
            # cv2.waitKey()

            logger.info("h_mult={}".format(h_mult))
            logger.info("w_mult={}".format(w_mult))

            # Perform inference on tiles of image for better accuracy
            for i in range(0, int(h_mult)):
                for j in range(0, int(w_mult)):
                    tile_np = image_pad_np[i*image_size:(i+1)*image_size, j*image_size:(j+1)*image_size, :]

                    logger.info("i={}, j={}".format(i, j))
                    # cv2.imshow('tile-i={}-j={}'.format(i, j), tile_np)
                    # cv2.waitKey()

                    tiles_np.append(tile_np)
                    tile_ins.append((i, j))

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            tiles_np_expanded = np.asarray(tiles_np)

            # Actual detection.
            output_dict = run_inference_for_multiple_images(tiles_np_expanded, detection_graph)

            # adjust the bounding box coordinates due to the tiling
            # for box in non_zero_detection_boxes:
            for tile_idx, tile_ins_tup in enumerate(tile_ins):
                i, j = tile_ins_tup[:]
                tile_np = tiles_np[tile_idx]
                boxes = output_dict['detection_boxes'][tile_idx]

                for box_idx, box in enumerate(boxes):
                    h_offset = i * image_size
                    w_offset = j * image_size
                    ymin, xmin, ymax, xmax = box
                    tile_h, tile_w, tile_d = tile_np.shape[:]

                    # the box coordinates were normalized so we must get the pixel value of each
                    ymin *= tile_h
                    ymax *= tile_h
                    xmin *= tile_w
                    xmax *= tile_w

                    # adjust the box coordinates to be relative to the original image instead of tile
                    ymin += h_offset
                    ymax += h_offset
                    xmin += w_offset
                    xmax += w_offset

                    # update boxes list
                    output_dict['detection_boxes'][tile_idx][box_idx][:] = [ymin, xmin, ymax, xmax]

                detection_classes.extend(output_dict['detection_classes'][tile_idx])
                detection_scores.extend(output_dict['detection_scores'][tile_idx])
                detection_boxes.extend(output_dict['detection_boxes'][tile_idx])

        # Visualization of the results of a detection.
        # This function groups boxes that correspond to the same location: https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.asarray(detection_boxes, dtype=np.float32),
            np.asarray(detection_classes, dtype=np.int64), # np arrays are double by nature, but this function requires ints for its classes
            np.asarray(detection_scores, dtype=np.float32),
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=False,  # we've adjusted the tiles' box coordinates
            line_thickness=8,
            min_score_thresh=0.1,
            max_boxes_to_draw=None) # None will force the function to look at all boxes in list which is what we want since our list of boxes isn't ordered in any way

        # save the original image with boxes
        basename = os.path.basename(image_path)[:-4] # get basename and remove extension of .png or .jpg
        out_image_np_path = "/home/nightrider/calacademy-fish-id/outputs/{}".format(basename)
        logger.info("tile_np_path={}".format(out_image_np_path))
        fu.save_images(images=[(out_image_np_path, image_np)])

        ## save the detection classes and scores to text file
        # First we grab only non-zero probability detection outputs
        # Note: we don't remove duplicate boxes as this could affect our evaluation metrics
        non_zero_outputs = np.asarray(detection_scores, dtype=np.float32) > 0
        non_zero_detection_classes = np.asarray(detection_classes, dtype=np.int64)[non_zero_outputs] # indexing must be done on np array and not list
        non_zero_detection_scores = np.asarray(detection_scores, dtype=np.float32)[non_zero_outputs] # indexing must be done on np array and not list
        non_zero_detection_boxes = np.asarray(detection_boxes, dtype=np.float32)[non_zero_outputs] # indexing must be done on np array and not list

        out_image_np_text_path = "/home/nightrider/calacademy-fish-id/outputs/{}.txt".format(basename)
        out_image_np_text = open(out_image_np_text_path, "a+")
        for pr_tuple in zip(non_zero_detection_classes, non_zero_detection_scores, non_zero_detection_boxes):
            pr_class = category_index[pr_tuple[0]]["name"]
            out_image_np_text.write("{} {} {}\n".format(pr_class, pr_tuple[1], " ".join(map(str, pr_tuple[2]))))
        out_image_np_text.close()


if __name__ == "__main__":
    # we expect, as a hand-shake agreement, that there is a .yml config file in top level of lib/configs directory
    config_dir = os.path.join(os.curdir, 'configs')
    yaml_path = os.path.join(config_dir, 'inference_pretrained_tf_obj_detect_model.yml')
    with open(yaml_path, "r") as stream:
        config = yaml.load(stream)

    ## collect hyper parameters/args from config
    # NOTE: float() is required to parse any exponentials since YAML sends exponentials as strings
    model_name = config["model_name"]
    download_base = config["download_base"]
    path_to_test_images_dir = config["path_to_test_images_dir"]
    input_image_sizes = config["input_image_sizes"]
    label_map = config["label_map"]

    # For the sake of simplicity we will use only 1 image:
    # image1.jpg
    # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
    test_image_paths = [os.path.join(path_to_test_images_dir, 'image{}.jpg'.format(i)) for i in
                        range(1, 2)]  # TODO: use lib/file_utils.py

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    model_file = model_name + '.tar.gz'
    path_to_frozen_graph = os.path.join(model_name, 'frozen_inference_graph.pb')

    # List of the strings that is used to add correct label for each box.
    path_to_labels = os.path.join('/home/nightrider/tensorflow/models/research/object_detection', 'data', label_map)
    category_index = label_map_util.create_category_index_from_labelmap(path_to_labels, use_display_name=True)

    # download model files
    opener = urllib.request.URLopener()
    opener.retrieve(download_base + model_file, model_file)
    tar_file = tarfile.open(model_file)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # predict_images_tiled()
    predict_images_batched(test_image_paths=test_image_paths,
                           input_image_sizes=input_image_sizes,
                           category_index=category_index)