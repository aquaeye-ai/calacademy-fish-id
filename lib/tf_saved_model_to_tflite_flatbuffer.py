###
# Attempts to use tutorial from: https://www.tensorflow.org/lite/convert/python_api#examples_

import tensorflow as tf

EXPORT_DIR = "/home/nightrider/Downloads/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18"

# Convert the model.
converter = tf.lite.TFLiteConverter.from_saved_model(EXPORT_DIR)
tflite_model = converter.convert()