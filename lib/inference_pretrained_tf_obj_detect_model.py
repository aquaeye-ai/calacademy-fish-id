import numpy as np
import cv2 as cv2
import file_utils as fu
import os
import sys
import tarfile
from PIL import Image
import tensorflow as tf
import six.moves.urllib as urllib
from matplotlib import pyplot as plt
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util


# What model to download.
MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/home/nightrider/tensorflow/models/research/object_detection', 'data', 'mscoco_complete_label_map.pbtxt')
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# List of the strings that is used to add correct label for each box.
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = "/home/nightrider/calacademy-fish-id/datasets/pcr/stills/full/test"
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 2)] #TODO: use lib/file_utils.py

# Size, in pixels of input image
IMAGE_H = IMAGE_W = [300, 500]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


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


if __name__ == "__main__":
    for image_path in TEST_IMAGE_PATHS:
        for k in range(0, len(IMAGE_H)):
            #image = Image.open(image_path)

            # The array based representation of the image will be used later in order to prepare the result image with
            # boxes and labels on it.
            #image_np = load_image_into_numpy_array(image)
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
            cv_img = cv2.imread(image_path)
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

                    # display bounding boxes
                    # plt.figure(figsize=(IMAGE_SIZE))
                    # # plt.imshow(tile_np)
                    # plt.imshow(image_np)
                    # while True:
                    #     if plt.waitforbuttonpress():
                    #         break

                    # save the tile with boxes
                    # basename = os.path.basename(image_path)[:-4] # get basename and remove extension of .png or .jpg
                    # tile_np_path = "/home/nightrider/calacademy-fish-id/outputs/{}_tile_{}_{}_{}x{}".format(basename, i, j, IMAGE_H[k], IMAGE_W[k])
                    # print("tile_np_path={}".format(tile_np_path))
                    # fu.save_images(images=[(tile_np_path, tile_np)])
                    #
                    # tile_np_text_path = "/home/nightrider/calacademy-fish-id/outputs/{}_tile_{}_{}_{}x{}".format(basename, i, j, IMAGE_H[k], IMAGE_W[k])
                    # tile_np_text = open(tile_np_text_path, "a+")
                    # tile_np_text.write("detection_classes_{}_detection_scores_{}".format(output_dict['detection_classes'], output_dict['detection_scores']))
                    # tile_np_text.close()

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
        # first we grab only valid detection outputs
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