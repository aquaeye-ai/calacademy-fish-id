# Adapted from: https://towardsdatascience.com/detailed-tutorial-build-your-custom-real-time-object-detector-5ade1017fd2d#d582

import os
import shutil
import tarfile
import urllib2

import file_utils as fu


# Some models to train on
MODELS_CONFIG = {
    'ssd_mobilenet_v2': {
        'model_name': 'ssd_mobilenet_v2_coco_2018_03_29',
    },
    'faster_rcnn_inception_v2': {
        'model_name': 'faster_rcnn_inception_v2_coco_2018_01_28',
    },
}

# Select a model from `MODELS_CONFIG`.
# I chose ssd_mobilenet_v2 for this project, you could choose any
selected_model = 'ssd_mobilenet_v2'

if __name__ == "__main__":
    # Name of the object detection model to use.
    model = MODELS_CONFIG[selected_model]['model_name']

    #the distination folder where the model will be saved
    #change this if you have a different working dir
    dest_dir_path = os.path.join('/home/nightrider/calacademy-fish-id/classifiers/models/pretrained', model)

    # create destination directory if none exists
    fu.init_directory(directory=dest_dir_path)

    #selecting the model
    model_file = model + '.tar.gz'

    #creating the downlaod link for the model selected
    download_base = 'http://download.tensorflow.org/models/object_detection/'

    # full model path
    model_file_path = os.path.join(dest_dir_path, model_file)

    # model download url
    model_url = os.path.join(download_base, model_file)

    #checks if the model has already been downloaded, download it otherwise
    if not (os.path.exists(model_file_path)):
        zip_file = urllib2.urlopen(model_url)
        with open(model_file_path, 'wb') as output:
            shutil.copyfileobj(zip_file, output)

    #unzipping the model and extracting its content
    tar = tarfile.open(model_file_path)
    tar.extractall()
    tar.close()

    # creating an output file to save the model while training
    os.remove(model_file_path)
    if (os.path.exists(dest_dir_path)):
        shutil.rmtree(dest_dir_path)
    os.rename(model, dest_dir_path)