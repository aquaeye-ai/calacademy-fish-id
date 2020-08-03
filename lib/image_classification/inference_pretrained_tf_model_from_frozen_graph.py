"""
Script to inference a TF model from frozen graph.
For use with pretrained TF Slim models (e.g. those used with lib/image_classification/inference_pretrained_tf_model_from_frozen_graph.py)
and those models fine-tuned/trained using  lib/image_classification/retrain_1_b_1.py.

NOTE: the log.txt will be output in a parent directory as compared to the output_dir provided to the script.

TODO: Adopt the early, one-time instantiation of TF tensors flow of script lib/image_classification_evaluate_tf_model_from_frozen_graph.py
TODO: so that the script doesn't slow down from continuous grafting of tensors to the graph throughout the for-loop inference.
"""


import os
import yaml
import time
import logging
import tarfile

import cv2 as cv2
import numpy as np
import tensorflow as tf
import lib.file_utils as fu
import six.moves.urllib as urllib
import lib.log_utils as log_utils

from nets import inception
from nets import inception_v3
from datasets import imagenet
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util

# configure logging
OUTPUT_DIR = "/home/nightrider/calacademy-fish-id/outputs"
log_utils.LOG_DIR = OUTPUT_DIR
log_utils.init_logging(file_name="inference_pretrained_tf_model_from_frozen_graph_log.txt")

logger = logging.getLogger(__name__)


def load_img(path_img):
    """
    Load an image to tensorflow.
    Adapted from: https://stackoverflow.com/questions/43341970/tensorflow-dramatic-loss-of-accuracy-after-freezing-graph

    :param path_img: image path on the disk
    :return: 3D tensorflow image
    """
    filename_queue = tf.train.string_input_producer([path_img])  # list of files to read

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)

    my_img = tf.image.decode_image(value)  # use png or jpg decoder based on your files.

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)

        # Start populating the filename queue.

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(1):  # length of your filename list
            image = my_img.eval()  # here is your image Tensor :)

        print(image.shape)
        # Image.fromarray(np.asarray(image)).show()

        coord.request_stop()
        coord.join(threads)

        return image

#TODO: consider replacing this functionality with cv2
def preprocess(image, height, width, central_fraction=0.875, scope=None, apply_tf_slim_preprocessing=False):
    """Prepare one image for evaluation.

    If height and width are specified it would output an image with that size by
    applying resize_bilinear.

    If central_fraction is specified it would cropt the central fraction of the
    input image.

    Adapted from: https://stackoverflow.com/questions/43341970/tensorflow-dramatic-loss-of-accuracy-after-freezing-graph

    Args:
      image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
        [0, 1], otherwise it would converted to tf.float32 assuming that the range
        is [0, MAX], where MAX is largest positive representable number for
        int(8/16/32) data type (see `tf.image.convert_image_dtype` for details)
      height: integer
      width: integer
      central_fraction: Optional Float, fraction of the image to crop.
      scope: Optional scope for name_scope.
    Returns:
      3-D float Tensor of prepared image.
    """

    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    if central_fraction and apply_tf_slim_preprocessing == True:
        image = tf.image.central_crop(image, central_fraction=central_fraction)

    if height and width:
        # Resize the image to the specified height and width.
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [height, width],
                                         align_corners=False)
        image = tf.squeeze(image, [0])

    if apply_tf_slim_preprocessing == True:
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)

    return image

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

@log_utils.timeit
def run_inference_for_multiple_images(images=None, graph=None, sess=None, output_node=None, input_node=None):
    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        output_node
    ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

    # we need to expand the first dimension if we only inference on one image, since the model inference expects a batch
    if len(images.shape) == 3:
        images = np.expand_dims(images, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name(input_node + ':0')

    # Run inference
    t1 = time.time()
    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: images})
    t2 = time.time()
    logger.info("inference time: {}".format(t2-t1))

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict[output_node] = output_dict[output_node][:]

    return output_dict

@log_utils.timeit
def predict_images_tiled_batched(test_image_paths=None, tile_sizes=None, category_index=None, min_score_threshold=None,
                                 model_input_size=None, output_node=None):
    for im_idx, image_path in enumerate(test_image_paths):
        logger.info("image: {}".format(image_path))

        for k, tile_size in enumerate(tile_sizes):
            logger.info("image size: {}x{}".format(tile_size, tile_size))

            tiles_np = []
            tiles_ins = []

            image = load_img(image_path)
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image_np = tf.Session().run(image)
            # tiles_np, tiles_ins, num_tiles = get_tiles_for_image(image=image, tile_size=tile_size)

            class_scores = []

            # Pad image dimensions to nearest multiple of tile_size so that we can operate on crops
            h_mult = np.ceil(image_np.shape[0] / float(tile_size))
            w_mult = np.ceil(image_np.shape[1] / float(tile_size))
            h_new = h_mult * tile_size
            w_new = w_mult * tile_size
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
            num_tiles = 0
            for i in range(0, int(h_mult)):
                for j in range(0, int(w_mult)):
                    tile_np = image_pad_np[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size, :]

                    logger.info("i={}, j={}".format(i, j))
                    # cv2.imshow('tile-i={}-j={}'.format(i, j), tile_np)
                    # cv2.waitKey()

                    tile_np = preprocess(tile_np, model_input_size, model_input_size)
                    tile_np = tf.Session().run(tile_np)

                    tiles_np.append(tile_np)
                    tiles_ins.append((i, j))
                    num_tiles += 1

            logger.info("num_tiles={}".format(num_tiles))

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            tiles_np_expanded = np.asarray(tiles_np)

            # Actual detection.
            output_dict = run_inference_for_multiple_images(tiles_np_expanded, detection_graph, sess=sess, output_node=output_node)

            class_scores = output_dict[output_node][:]

            # sort the class_scores
            sorted_class_scores = [sorted(enumerate(class_scores[idx]), key=lambda x: x[1], reverse=True) for idx, x in enumerate(class_scores[:])]

            ## save the detection classes and scores to text file
            # First we threshold detection outputs.
            thresh_outputs = np.asarray(sorted_class_scores, dtype=np.float32)[:, :, 1] > min_score_threshold
            thresh_class_scores = []
            for tile_idx, tile_classes in enumerate(thresh_outputs):
                thresh_class_scores.append([])
                for class_idx, tile_class in enumerate(tile_classes):
                    if tile_class == True:
                        thresh_class_scores[tile_idx].append(sorted_class_scores[tile_idx][class_idx])
            thresh_class_names = []
            for tile_idx, tile_classes in enumerate(thresh_class_scores):
                thresh_class_names.append([])
                for class_idx, class_tuple in enumerate(tile_classes):
                    thresh_class_names[tile_idx].append(category_index[class_tuple[0]])

            out_image_np_text_path = os.path.join(OUTPUT_DIR, "{}.txt".format(os.path.basename(image_path[:-4])))
            out_image_np_text = open(out_image_np_text_path, "a+")
            out_image_np_text.write("predictions for tile size: {}x{}\n\n".format(tile_size, tile_size))
            for tile_idx, pr_tuple in enumerate(zip(thresh_class_names, thresh_class_scores)):
                out_image_np_text.write("tile {}x{}: {} {}\n".format(tiles_ins[tile_idx][0], tiles_ins[tile_idx][1], pr_tuple[0], pr_tuple[1], " ".join(map(str, pr_tuple[1]))))
            out_image_np_text.close()

def predict_images_whole(test_image_paths=None, category_index=None, min_score_threshold=None, model_input_size=None,
                         output_node=None, input_node=None, apply_tf_slim_preprocessing=False):
    """
    Inferences model on entire image for each image path supplied.

    :param test_image_paths: list of strings, paths to images
    :param category_index: dictionary, key=class index in output and value=class name (string)
    :param min_score_threshold: float, minimum prediction confidence to accept in the output
    :param model_input_size: int, taken as nxn dimensions of model input
    :return: None
    """
    for im_idx, image_path in enumerate(test_image_paths):
        logger.info("image: {}".format(image_path))

        image = load_img(image_path)
        image = preprocess(image, model_input_size, model_input_size, apply_tf_slim_preprocessing=apply_tf_slim_preprocessing)
        image_np = tf.Session().run(image)

        h, w = image_np.shape[:2]
        logger.info("image size: {}x{}".format(h, w))

        # cv2.imshow('image_np', image_np)
        # cv2.waitKey()

        ## Actual detection.
        # Both of these produce the same but I use Reshape_1 to stay in line with tf slim's tutorial: https://github.com/tensorflow/models/tree/master/research/slim#Export
        # output_node = 'InceptionV3/Predictions/Softmax'
        output_dict = run_inference_for_multiple_images(image_np, detection_graph, sess=sess, output_node=output_node,
                                                        input_node=input_node)

        class_scores = output_dict[output_node][0]

        # sort the class_scores
        sorted_class_scores = sorted(enumerate(class_scores), key=lambda x: x[1], reverse=True)

        ## save the detection classes and scores to text file
        # First we threshold detection outputs.
        thresh_outputs = np.asarray(sorted_class_scores, dtype=np.float32)[:, 1] >  min_score_threshold
        thresh_class_scores = [sorted_class_scores[idx] for idx, x in enumerate(thresh_outputs) if x == True]
        thresh_class_names = [category_index[x[0]] for x in thresh_class_scores]

        out_image_np_text_path = os.path.join(OUTPUT_DIR, "{}.txt".format(os.path.basename(image_path[:-4])))
        out_image_np_text = open(out_image_np_text_path, "a+")
        for pr_tuple in zip(thresh_class_names, thresh_class_scores):
            out_image_np_text.write("{} {}\n".format(pr_tuple[0], pr_tuple[1][1], " ".join(map(str, pr_tuple[1]))))
        out_image_np_text.close()


if __name__ == "__main__":
    # we expect, as a hand-shake agreement, that there is a .yml config file in top level of lib/configs directory
    config_dir = os.path.join(os.curdir, '..', 'configs', 'image_classification')
    yaml_path = os.path.join(config_dir, 'inference_pretrained_tf_model_from_frozen_graph.yml')
    with open(yaml_path, "r") as stream:
        config = yaml.load(stream)

    ## collect hyper parameters/args from config
    # NOTE: float() is required to parse any exponentials since YAML sends exponentials as strings
    # model_name = config["model_name"]
    # download_base = config["download_base"]
    path_to_test_images_dir = config["path_to_test_images_dir"]
    tile_sizes = config["tile_sizes"]
    model_input_size = config["model_input_size"]
    min_score_threshold = config["min_score_threshold"]
    path_to_frozen_graph = config["path_to_frozen_graph"]
    path_to_labels = config["path_to_labels"]
    output_node = config["output_node"]
    input_node = config["input_node"]
    use_imagenet_labels = False if config["use_imagenet_labels"] <= 0 else True
    apply_tf_slim_preprocessing = False if config["apply_tf_slim_preprocessing"] <= 0 else True

    # grab image paths
    test_image_paths = fu.find_images(directory=path_to_test_images_dir, extension=".jpg")

    # Dictionary of the strings that is used to add correct label for each class index in the model's output.
    # key: index in output
    # value: string name of class
    # TODO: we could likely collapse this if statement if we spent the time to look up the file for imagenet (and consequently not need the use_imagenet_labels flag)
    category_index = {}
    if use_imagenet_labels == True:
        category_index = imagenet.create_readable_names_for_imagenet_labels()
    else:
        with open(path_to_labels) as labels_f:
            for idx, line in enumerate(labels_f):
                category_index[idx] = line.strip()

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        with tf.Session() as sess:
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

                # predict_images_tiled_batched(test_image_paths=test_image_paths,
                #                              tile_sizes=tile_sizes,
                #                              category_index=category_index,
                #                              min_score_threshold=min_score_threshold,
                #                              model_input_size=model_input_size,
                #                              output_node=output_node)
                predict_images_whole(test_image_paths=test_image_paths,
                                     category_index=category_index,
                                     min_score_threshold=min_score_threshold,
                                     model_input_size=model_input_size,
                                     output_node=output_node,
                                     input_node=input_node,
                                     apply_tf_slim_preprocessing=apply_tf_slim_preprocessing)