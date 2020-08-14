"""
Script to view livestream, e.g. youtube link for https://research.calacademy.org/learn-explore/animal-webcams/reef-lagoon-cam
"""

import os
import cv2
import yaml
import logging


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # we expect, as a hand-shake agreement, that there is a .yml config file in top level of lib/configs directory
    config_dir = os.path.join(os.curdir, 'configs')
    yaml_path = os.path.join(config_dir, 'view_livestream.yml')
    with open(yaml_path, "r") as stream:
        config = yaml.load(stream)

    ## collect hyper parameters/args from config
    # NOTE: float() is required to parse any exponentials since YAML sends exponentials as strings
    stream_url = config["stream_url"]
    dst_dir = config["destination_directory"]

    cap = cv2.VideoCapture(stream_url)

    # set limited buffer size so that we always have the latest frame: https://stackoverflow.com/questions/58293187/opencv-real-time-streaming-video-capture-is-slow-how-to-drop-frames-or-get-sync
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    if not cap:
        logger.info("Failed VideoCapture")

    while (True):
        _, frame = cap.read()
        cv2.imshow("Camera Capture", frame)

        # try to match delay with FPS so as to try to read at the speed of the video feed: https://stackoverflow.com/questions/58293187/opencv-real-time-streaming-video-capture-is-slow-how-to-drop-frames-or-get-sync
        # we guess that the FPS of the video feed is 30fps
        fps = 1. /30
        delay_ms = fps * 1000
        cv2.waitKey(int(delay_ms))