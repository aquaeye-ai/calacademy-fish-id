"""
Script to evaluate and provide Top-1 and Top-K stats on test data for a TF model from frozen graph.
For use with keras-saved hdf5-formatted pretrained TF models and those models fine-tuned/trained using
lib/image_classification/retrain_3_b_i.py.

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
import shutil
import logging
import tarfile

import cv2 as cv2
import numpy as np
import tensorflow as tf
import lib.scripts.file_utils as fu
import six.moves.urllib as urllib
import lib.scripts.log_utils as log_utils

from tensorflow import keras
print("TensorFlow version is ", tf.__version__)

from nets import inception
from nets import inception_v3
from datasets import imagenet
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util


# configure logging
OUTPUT_DIR = "/home/nightrider/aquaeye-ai/calacademy-fish-id/outputs/image_classification"
log_utils.LOG_DIR = OUTPUT_DIR
log_utils.init_logging(file_name="inference_trained_tf_model_from_keras_log.txt")

logger = logging.getLogger(__name__)


@log_utils.timeit
def predict_images_whole(test_image_paths=None, category_index=None, K=None, model=None):
    """
    Inferences model on entire image for each image path supplied.

    :param test_image_paths: list of strings, paths to images
    :param category_index: dictionary, key=class index in output and value=class name (string)
    :param K: int, K value for Top-K stats
    :return: list of tuples where each tuple = (image path, class name, sorted-descending list of Top-K class predictions)
    """
    results = []
    for im_idx, image_path in enumerate(test_image_paths):
        logger.info("{}/{} image: {}".format(im_idx+1, len(test_image_paths), image_path))

        image_np = cv2.imread(image_path)

        # cv2 loads image in BGR format but model needs RGB format
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        # scale image since we trained on scaled images
        image_np = image_np * (1. / 255)

        h, w = image_np.shape[:2]
        logger.info("image size: {}x{}".format(h, w))

        # cv2.imshow('image_np', image_np)
        # cv2.waitKey()

        ## Actual detection.

        image_np = cv2.resize(image_np, dsize=(model_input_size, model_input_size), interpolation=cv2.INTER_LINEAR)
        image_np = np.expand_dims(image_np, axis=0)
        output_dict = model.predict(image_np)


        class_scores = output_dict[0]

        # sort the class_scores
        top_k_class_scores_idx = np.argsort(class_scores)[-K:]
        top_k_class_scores_idx = list(reversed(top_k_class_scores_idx))
        top_k_class_scores = class_scores[top_k_class_scores_idx]


        ## return the top-k classes and scores to text file
        class_names = [category_index[i] for i in top_k_class_scores_idx]

        results.append((image_path, class_names, top_k_class_scores))

    return results


if __name__ == "__main__":
    # we expect, as a hand-shake agreement, that there is a .yml config file in top level of lib/configs directory
    config_dir = os.path.join('..', '..', 'configs', 'image_classification')
    yaml_path = os.path.join(config_dir, 'evaluate_tf_model_from_keras_h5.yml')
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
    path_to_model_h5 = config["path_to_model_h5"]

    path_to_labels = config["path_to_labels"]
    use_imagenet_labels = False if config["use_imagenet_labels"] <= 0 else True
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
        category_index = imagenet.create_readable_names_for_imagenet_labels() #TODO this may need sorting
    else:
        with open(path_to_labels) as labels_f:
            # we need to sort the labels since this is how keras reads the labels in during training: https://stackoverflow.com/questions/38971293/get-class-labels-from-keras-functional-model
            idx = 0
            for line in sorted(labels_f):
                category_index[idx] = line.strip()
                idx += 1

    # load model
    model = keras.models.load_model(path_to_model_h5)

    with open(os.path.join(output_dir, 'stats.txt'), 'w+') as stats_f, open(os.path.join(output_dir, 'misclassifications_top_1.txt'), 'w+') \
            as misclassifications_top_1_f, open(os.path.join(output_dir, 'misclassifications_top_k.txt'), 'w+') \
            as misclassifications_top_k_f:

                t1 = time.time()
                total_num_images = 0
                total_num_correct_top_1 = 0
                total_num_correct_top_k = 0
                total_accuracy_top_1 = 0.0
                total_accuracy_top_k = 0.0
                for cls_idx, class_dir in enumerate(class_dirs):
                    logger.info("{}/{} Class: {}".format(cls_idx, len(class_dirs), class_dir))

                    # init directories to contain Top-1 misclassified images for class
                    top_1_incorrect_img_dir = os.path.join(output_dir, 'images', 'top_1_incorrect_images', class_dir)
                    fu.init_directory(directory=top_1_incorrect_img_dir)

                    class_image_paths = fu.find_images(directory=os.path.join(path_to_test_images_dir, class_dir),
                                                       extension=".jpg")
                    if len(class_image_paths) > 0: # if we don't have examples for class then guard against division by zero and don't report stats for class
                        total_num_images += len(class_image_paths)

                        results = predict_images_whole(test_image_paths=class_image_paths,
                                                       category_index=category_index,
                                                       K=K,
                                                       model=model)

                        # gather Top-1 stats
                        num_incorrect_top_1 = 0
                        image_paths_incorrect_top_1 = []
                        incorrect_class_votes_dict = {}
                        for r_idx, result in enumerate(results):
                            if result[1][0] != class_dir:
                                if result[1][0] in incorrect_class_votes_dict:
                                    incorrect_class_votes_dict[result[1][0]] += 1
                                else:
                                    incorrect_class_votes_dict[result[1][0]] = 1
                                num_incorrect_top_1 += 1

                                # provide the tuples of (image path, Top-K most common incorrect labels, probabilities of top-K most common incorrect labels)
                                image_paths_incorrect_top_1.append((result[0], result[1][:K], result[2][:K]))

                                # copy image to misclassified directory for later viewing
                                dst_mis_img_name = "{}_{}_{}.jpg".format(result[1][0], result[2][0], r_idx)
                                dst_mis_img_path = os.path.join(top_1_incorrect_img_dir, dst_mis_img_name)
                                shutil.copy(result[0], dst_mis_img_path)
                            else:
                                total_num_correct_top_1 += 1


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
                            else:
                                total_num_correct_top_k += 1

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

                # write overall top-1 and top-K accuracy across all test examples
                stats_f.write("\nOverall Top-1 Accuracy: {:.2f}%\n".format(
                    (np.float(total_num_correct_top_1) / np.float(total_num_images)) * 100))
                stats_f.write("Overall Top-{} Accuracy: {:.2f}%".format(
                    K, (np.float(total_num_correct_top_k) / np.float(total_num_images)) * 100))

                t2 = time.time()
                logger.info("Total Images: {}".format(total_num_images))
                logger.info("Total Evaluation Time: {}".format(t2 - t1))