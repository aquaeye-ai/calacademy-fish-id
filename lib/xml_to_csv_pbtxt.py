# Adapted from: https://towardsdatascience.com/detailed-tutorial-build-your-custom-real-time-object-detector-5ade1017fd2d#d582
from __future__ import division, print_function, absolute_import

import os
import glob

import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    classes_names = []
    xml_list = []

    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.find('filename').text

        # If the .xml annotations were produced using an annotation tiling software such as: https://github.com/jdcast/image_bbox_slicer
        # then the filename with the suffix of .jpg won't be saved in the .xml file and we need to add it.  Otherwise
        # lib/to_TFRecord.py will fail.
        if not filename.lower().endswith('.jpg'):
            filename += '.jpg'

        for member in root.findall('object'):
            classes_names.append(member[0].text)
            value = (filename,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text))
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    classes_names = list(set(classes_names))
    classes_names.sort()
    return xml_df, classes_names

if __name__ == "__main__":
    data_dir_path = "/home/nightrider/calacademy-fish-id/datasets/pcr/stills/dry_run/crops/combined_300_600"

    for label_path in ['train_labels', 'test_labels']:
        image_path = os.path.join(data_dir_path, label_path)
        xml_df, classes = xml_to_csv(image_path)
        xml_df.to_csv("{}.csv".format(image_path), index=None)
        print("Successfully converted {} xml to csv.".format(image_path))

    label_map_path = os.path.join(data_dir_path, "label_map.pbtxt")
    pbtxt_content = ""

    for i, class_name in enumerate(classes):
        pbtxt_content += "item {{\n    id: {0}\n    name: '{1}'\n}}\n\n".format(i + 1, class_name)
    pbtxt_content = pbtxt_content.strip()
    with open(label_map_path, "w") as f:
        f.write(pbtxt_content)