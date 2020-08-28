"""
Script to record livestream, e.g. youtube link for https://research.calacademy.org/learn-explore/animal-webcams/reef-lagoon-cam
"""

import os
import sys
import yaml
import time
import logging
import subprocess


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # we expect, as a hand-shake agreement, that there is a .yml config file in top level of lib/configs directory
    config_dir = os.path.join(os.curdir, 'configs')
    yaml_path = os.path.join(config_dir, 'record_livestream.yml')
    with open(yaml_path, "r") as stream:
        config = yaml.load(stream)

    ## collect hyper parameters/args from config
    # NOTE: float() is required to parse any exponentials since YAML sends exponentials as strings
    stream_url = config["stream_url"]
    dst_dir = config["destination_directory"]
    record_time = config["record_time"]

    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    out_path_ts = os.path.join(dst_dir, "{}_length_{}s.ts".format(timestamp, record_time))
    out_path_mp4 = os.path.join(dst_dir, "{}_length_{}s.mp4".format(timestamp, record_time))

    # record stream in .ts format
    p1 = subprocess.Popen(['ffmpeg', '-i', '{}'.format(stream_url), '-c', 'copy', '{}'.format(out_path_ts)])

    # record for specified time
    time.sleep(record_time)
    p1.kill()

    # re-mux saved .ts file into .mp4
    subprocess.call(['ffmpeg', '-i', '{}'.format(out_path_ts), '-bsf:a', 'aac_adtstoasc', '-acodec', 'copy', '-vcodec', 'copy', '{}'.format(out_path_mp4)])

    # remove .ts file
    os.remove(out_path_ts)

    sys.exit(0)