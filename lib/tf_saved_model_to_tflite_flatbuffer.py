###
# Attempts to use tutorial from: https://www.tensorflow.org/lite/convert/python_api#examples_

import tensorflow as tf

EXPORT_DIR = "/home/nightrider/Downloads/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18"

# with tf.Session() as sess:
#     builder = tf.saved_model.builder.SavedModelBuilder(EXPORT_DIR)
#     builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],)
#     builder.save()

# Convert the model.
converter = tf.lite.TFLiteConverter.from_saved_model(EXPORT_DIR)
tflite_model = converter.convert()