# Script used to sample video file for frames to annotate.

import os
import cv2
import yaml

if __name__ == "__main__":
    # we expect, as a hand-shake agreement, that there is a .yml config file in top level of lib/configs directory
    config_dir = os.path.join(os.curdir, 'configs')
    yaml_path = os.path.join(config_dir, 'sample_video_frames.yml')
    with open(yaml_path, "r") as stream:
        config = yaml.load(stream)

    ## collect hyper parameters/args from config
    # NOTE: float() is required to parse any exponentials since YAML sends exponentials as strings
    sample_rate = config["sample_rate"]
    video_path = config["video_path"]
    destination_directory = config["destination_directory"]

    basename_vid = os.path.basename(video_path)

    vidcap = cv2.VideoCapture(video_path)
    success, frame = vidcap.read()
    count = 0

    while success:
        if count % sample_rate == 0:
            dest_img_name = "video_{}_frame_{}.jpg".format(basename_vid[:-4], count) # remove extension from video name
            dest_image_path = os.path.join(destination_directory, dest_img_name)
            cv2.imwrite(dest_image_path, frame)  # save frame as JPEG file

        # get the next frame
        success, frame = vidcap.read()
        print('Read a new frame: ', success)
        count += 1