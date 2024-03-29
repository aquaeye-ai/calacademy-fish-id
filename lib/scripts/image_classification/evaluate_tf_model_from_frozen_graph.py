"""
Script to evaluate and provide Top-1 and Top-K stats on test data for a TF model from frozen graph.
For use with pretrained TF Slim models (e.g. those used with lib/image_classification/inference_pretrained_tf_model_from_frozen_graph.py)
and those models fine-tuned/trained using  lib/image_classification/retrain_1_b_i.py.

NOTE: the log.txt will be output in a parent directory as compared to the output_dir provided to the script.

Stats produced by script:
1) Top-1 Accuracy per class
2) Top-K Accuracy per class
3) Top-K most common incorrect labels per class with frequency occurrence counts
4) Total number of images evaluated per class
5) Average Top-1 Accuracy across all classes
6) Average Top-K Accuracy across all classes
7) List of images organized by class whose Top-1 prediction was incorrect, along with their Top-K most common labels/probabilities
8) List of images organized by class whose Top-K predictions were all incorrect, along with their Top-K most common labels/probabilities
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
OUTPUT_DIR = "/home/nightrider/calacademy-fish-id/outputs/image_classification"
log_utils.LOG_DIR = OUTPUT_DIR
log_utils.init_logging(file_name="inference_pretrained_tf_model_from_frozen_graph_log.txt")

logger = logging.getLogger(__name__)


def def_load_img_op(path_img):
    """
    Load an image to tensorflow.
    Adapted from: https://stackoverflow.com/questions/43341970/tensorflow-dramatic-loss-of-accuracy-after-freezing-graph

    :param path_img: image path on the disk
    :return: 3D tensorflow image
    """
    img_data = tf.io.read_file(path_img)
    image_op = tf.image.decode_image(img_data)  # use png or jpg decoder based on your files.

    return image_op

#TODO: consider replacing this functionality with cv2
def def_preprocess_ops(image, height, width, central_fraction=0.875, scope=None, apply_tf_slim_preprocessing=False):
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

    image_data = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    if central_fraction and apply_tf_slim_preprocessing == True:
        image_data = tf.image.central_crop(image_data, central_fraction=central_fraction)

    if height and width:
        # Resize the image to the specified height and width.
        image_data = tf.expand_dims(image_data, 0)
        image_data = tf.image.resize_bilinear(image_data, [height, width], align_corners=False)
        image_data = tf.squeeze(image_data, [0])

    if apply_tf_slim_preprocessing == True:
        image_data = tf.subtract(image_data, 0.5)
        image_data = tf.multiply(image_data, 2.0)

    return image_data

@log_utils.timeit
def run_inference_for_multiple_images(images=None, output_node=None, input_tensor=None, tensor_dict=None):
    """
    Call sess.run on full graph for inference of batch of images (can be single image).

    :param images: numpy array of images
    :param output_node: str, name of output node in graph
    :param input_tensor: str, name of input node in graph
    :param tensor_dict: dict of tf tensors and arguments to supply session.run() call with
    :return: dict of outputs from inference
    """
    # we need to expand the first dimension if we only inference on one image, since the model inference expects a batch
    if len(images.shape) == 3:
        images = np.expand_dims(images, 0)

    # Run inference
    t1 = time.time()
    output_dict = sess.run(tensor_dict, feed_dict={input_tensor: images})
    t2 = time.time()
    logger.info("inference time: {}".format(t2-t1))

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict[output_node] = output_dict[output_node][:]

    return output_dict

def predict_images_whole(test_image_paths=None, category_index=None, output_node=None, input_node=None, K=None,
                         preprocess_op=None, image_placeholder=None, input_tensor=None, tensor_dict=None,
                         load_img_op=None, image_path_placeholder=None):
    """
    Inferences model on entire image for each image path supplied.

    :param test_image_paths: list of strings, paths to images
    :param category_index: dictionary, key=class index in output and value=class name (string)
    :param output_node: str, name of output node in graph
    :param input_node: str, name of input node in graph
    :param K: int, K value for Top-K stats
    :param preprocess_op: tf op to apply for preprocessing
    :param image_placeholder: tf placeholder for image
    :param input_tensor: tf tensor for graph input
    :param tensor_dict: tensor dict arg for sess.run of full graph
    :param load_img_op: tf op to apply to load image
    :param image_path_placeholder: tf placeholder to hold path to image
    :return: list of tuples where each tuple = (image path, class name, sorted-descending list of Top-K class predictions)
    """
    results = []
    for im_idx, image_path in enumerate(test_image_paths):
        logger.info("{}/{} image: {}".format(im_idx+1, len(test_image_paths), image_path))

        image = tf.Session().run(load_img_op, feed_dict={image_path_placeholder: image_path})
        image_np = tf.Session().run(preprocess_op, feed_dict={image_placeholder: image})

        h, w = image_np.shape[:2]
        logger.info("image size: {}x{}".format(h, w))

        # cv2.imshow('image_np', image_np)
        # cv2.waitKey()

        ## Actual detection.
        # Both of these produce the same but I use Reshape_1 to stay in line with tf slim's tutorial: https://github.com/tensorflow/models/tree/master/research/slim#Export
        # output_node = 'InceptionV3/Predictions/Softmax'
        output_dict = run_inference_for_multiple_images(images=image_np, output_node=output_node,
                                                        input_tensor=input_tensor, tensor_dict=tensor_dict)

        class_scores = output_dict[output_node][0]

        # sort the class_scores
        sorted_class_scores = sorted(enumerate(class_scores), key=lambda x: x[1], reverse=True)

        ## return the top-k classes and scores to text file
        class_names = [category_index[sorted_class_scores[i][0]] for i in range(K)]

        results.append((image_path, class_names, [sorted_class_scores[i][1] for i in range(K)]))

    return results


if __name__ == "__main__":
    # we expect, as a hand-shake agreement, that there is a .yml config file in top level of lib/configs directory
    config_dir = os.path.join(os.curdir, '..', 'configs', 'image_classification')
    yaml_path = os.path.join(config_dir, 'evaluate_tf_model_from_frozen_graph.yml')
    with open(yaml_path, "r") as stream:
        config = yaml.load(stream)

    ## collect hyper parameters/args from config
    # NOTE: float() is required to parse any exponentials since YAML sends exponentials as strings
    # model_name = config["model_name"]
    # download_base = config["download_base"]
    path_to_test_images_dir = config["path_to_test_images_dir"]
    output_dir = config["output_dir"]
    model_input_size = config["model_input_size"]
    min_score_threshold = config["min_score_threshold"]
    path_to_frozen_graph = config["path_to_frozen_graph"]
    path_to_labels = config["path_to_labels"]
    output_node = config["output_node"]
    input_node = config["input_node"]
    use_imagenet_labels = False if config["use_imagenet_labels"] <= 0 else True
    apply_tf_slim_preprocessing = False if config["apply_tf_slim_preprocessing"] <= 0 else True
    K = config["K"]

    # initialize output directory
    fu.init_directory(directory=output_dir)

    # get class directories
    class_image_paths = {}
    class_dirs = [d for d in os.listdir(path_to_test_images_dir)]

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

    # begin instantiating graph and ops
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        with tf.Session() as sess:
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

                ## Define various ops here, outside of the for loops...otherwise graph will append new op
                # for each call of these functions and grow out of control: https://stackoverflow.com/questions/44669869/tensorflows-computation-time-gradually-slowing-down-in-very-simple-for-loop

                # define image loading op
                image_path_placeholder = tf.placeholder(tf.string)
                load_img_op = def_load_img_op(image_path_placeholder)

                # define the image preprocessing ops
                image_placeholder = tf.placeholder(tf.uint8, shape=[None, None, 3])
                preprocess_op = def_preprocess_ops(image_placeholder, model_input_size, model_input_size,
                                   apply_tf_slim_preprocessing=apply_tf_slim_preprocessing)

                # get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    output_node
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

                input_tensor = tf.get_default_graph().get_tensor_by_name(input_node + ':0')

                with open(os.path.join(output_dir, 'stats.txt'), 'w+') as stats_f, open(os.path.join(output_dir, 'misclassifications_top_1.txt'), 'w+') \
                        as misclassifications_top_1_f, open(os.path.join(output_dir, 'misclassifications_top_k.txt'), 'w+') \
                        as misclassifications_top_k_f:

                            t1 = time.time()
                            total_num_images = 0
                            total_accuracy_top_1 = 0.0
                            total_accuracy_top_k = 0.0
                            for cls_idx, class_dir in enumerate(class_dirs):
                                logger.info("{}/{} Class: {}".format(cls_idx, len(class_dirs), class_dir))
                                class_image_paths = fu.find_images(directory=os.path.join(path_to_test_images_dir, class_dir),
                                                                   extension=".jpg")
                                total_num_images += len(class_image_paths)

                                results = predict_images_whole(test_image_paths=class_image_paths,
                                                               category_index=category_index,
                                                               output_node=output_node,
                                                               input_node=input_node,
                                                               K=K,
                                                               preprocess_op=preprocess_op,
                                                               image_placeholder=image_placeholder,
                                                               input_tensor=input_tensor,
                                                               tensor_dict=tensor_dict,
                                                               load_img_op=load_img_op,
                                                               image_path_placeholder=image_path_placeholder)

                                # gather Top-1 stats
                                num_incorrect_top_1 = 0
                                image_paths_incorrect_top_1 = []
                                incorrect_class_votes_dict = {}
                                for result in results:
                                    if result[1][0] != class_dir:
                                        if result[1][0] in incorrect_class_votes_dict:
                                            incorrect_class_votes_dict[result[1][0]] += 1
                                        else:
                                            incorrect_class_votes_dict[result[1][0]] = 1
                                        num_incorrect_top_1 += 1

                                        # provide the tuples of (image path, Top-K most common incorrect labels, probabilities of top-K most common incorrect labels)
                                        image_paths_incorrect_top_1.append((result[0], result[1][:K], result[2][:K]))

                                percent_incorrect_top_1 = (np.float(num_incorrect_top_1) / np.float(len(results))) * 100.0
                                total_accuracy_top_1 += (100.0 - percent_incorrect_top_1)

                                max_num_votes = 0
                                incorrect_class_votes_list = [(key, value) for key, value in incorrect_class_votes_dict.items()]
                                incorrect_class_votes_list_sorted = sorted(incorrect_class_votes_list, key=lambda tup: tup[1], reverse=True)

                                # gather Top-K stats
                                num_incorrect_top_k = 0
                                image_paths_incorrect_top_k = []
                                for result in results:
                                    if class_dir not in result[1]:
                                        num_incorrect_top_k += 1

                                        # provide the tuples of (image path, Top-K most common incorrect labels, probabilities of top-K most common incorrect labels)
                                        image_paths_incorrect_top_k.append((result[0], result[1][:K], result[2][:K]))

                                percent_incorrect_top_k = (np.float(num_incorrect_top_k) / np.float(len(results))) * 100.0
                                total_accuracy_top_k += (100.0 - percent_incorrect_top_k)

                                ## write stats to file

                                # gather Top-K most common, incorrect labels for class
                                incorrect_labels_and_probs_class_top_k = None
                                if len(incorrect_class_votes_list_sorted) >= K:  # we may have fewer than K incorrect labels if this class performed well
                                    incorrect_labels_and_probs_class_top_k = str([': '.join(map(str, tup)) for tup in incorrect_class_votes_list_sorted[:K]])
                                else:
                                    incorrect_labels_and_probs_class_top_k = str([': '.join(map(str, tup)) for tup in incorrect_class_votes_list_sorted[:]])

                                # write general stats
                                stats_f.write("{}:\n \tTop-1 Accuracy = {:.2f}%, Top-{} Accuracy: {:.2f}%, "
                                              "Top-{} Incorrect Labels: {}, "
                                              "Total Images: {}\n".format(class_dir,
                                                                          100.0 - percent_incorrect_top_1, K,
                                                                          100.0 - percent_incorrect_top_k, K,
                                                                          incorrect_labels_and_probs_class_top_k,
                                                                          len(class_image_paths)))
                                stats_f.flush()

                                # write predictions stats for images whose true class didn't match their Top-1 prediction
                                misclassifications_top_1_f.write("{}:\n".format(class_dir))
                                for mis in image_paths_incorrect_top_1:
                                    # write filename
                                    misclassifications_top_1_f.write("\t{}\n".format(os.path.basename(mis[0])))

                                    # write top-K most common labels/probabilities
                                    misclassifications_top_1_f.write("\t\t{}\n".format(
                                        [': '.join(map(str, tup)) for tup in zip(mis[1][:], mis[2][:])]))
                                    misclassifications_top_1_f.flush()

                                # write predictions stats for images whose true class didn't match their Top-K predictions
                                misclassifications_top_k_f.write("{}:\n".format(class_dir))
                                for mis in image_paths_incorrect_top_k:
                                    # write filename
                                    misclassifications_top_k_f.write("\t{}\n".format(os.path.basename(mis[0])))

                                    # write top-K most common labels/probabilities
                                    misclassifications_top_k_f.write("\t\t{}\n".format(
                                        [': '.join(map(str, tup)) for tup in zip(mis[1][:], mis[2][:])]))
                                    misclassifications_top_k_f.flush()

                            # write average Top-1 and Top-K accuracies across classes
                            stats_f.write("\nAverage Top-1 Accuracy: {:.2f}%\n".format(
                                total_accuracy_top_1 / np.float(len(class_dirs))))
                            stats_f.write("Average Top-{} Accuracy: {:.2f}%".format(
                                K, total_accuracy_top_k / np.float(len(class_dirs))))

                            t2 = time.time()
                            logger.info("Total Images: {}".format(total_num_images))
                            logger.info("Total Evaluation Time: {}".format(t2 - t1))