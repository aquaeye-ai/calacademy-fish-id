# Adapted from: https://towardsdatascience.com/detailed-tutorial-build-your-custom-real-time-object-detector-5ade1017fd2d#d582
from __future__ import division, print_function, absolute_import

from PIL import Image
from collections import namedtuple
from object_detection.utils import dataset_util

import os
import io

import pandas as pd
import tensorflow as tf


# change this to the base directory where your data/ is
DATA_BASE_URL = '/media/nightrider/Linux_2TB_HDD_A/my_datasets/calacademy/datasets/object_detection/reef_lagoon/stills/full/temp'

# location of images
IMAGE_DIR = os.path.join(DATA_BASE_URL, 'images/')


# def class_text_to_int(row_label):
#     if row_label == 'acanthurus blochii':
#         return 1
#     elif row_label == 'caesio teres':
#         return 2
#     elif row_label == 'heniochus diphreutes':
#         return 3
#     elif row_label == 'naso brevirostris':
#         return 4
#     elif row_label == 'other':
#         return 5
#     else:
#         None

def class_text_to_int(row_label):
    if row_label == 'other':
        return 1
    elif row_label == 'caesio teres':
        return 2
    else:
        None

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def create_tf_example(group, path):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


if __name__ == "__main__":
    # creates tfrecord for both csv's
    for csv in ['train_labels', 'test_labels']:
        input_path = os.path.join(DATA_BASE_URL, csv + '.csv')
        output_path = os.path.join(DATA_BASE_URL, csv + '.record')
        writer = tf.io.TFRecordWriter(output_path)
        examples = pd.read_csv(input_path, converters={'filename': lambda x: str(x)}) # converters necessary to preserve leading zeros in filenames (if leading zeros exist)
        grouped = split(examples, 'filename')

        for group in grouped:
            tf_example = create_tf_example(group, IMAGE_DIR)
            writer.write(tf_example.SerializeToString())

        writer.close()
        print('Successfully created the TFRecords: {}'.format(output_path))