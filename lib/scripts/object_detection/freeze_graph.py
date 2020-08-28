"""
Script that wraps tensorflow's freeze_graph functionality so that it can be run as demonstrated here:
https://towardsdatascience.com/detailed-tutorial-build-your-custom-real-time-object-detector-5ade1017fd2d#b948
"""
import os
import yaml
import subprocess

# from tensorflow.python.tools import freeze_graph

if __name__ == "__main__":
    # we expect, as a hand-shake agreement, that there is a .yml config file in top level of lib/configs directory
    config_dir = os.path.join(os.curdir, '..', 'configs', 'object_detection')
    yaml_path = os.path.join(config_dir, 'freeze_graph.yml')
    with open(yaml_path, "r") as stream:
        config = yaml.load(stream)

    ## collect hyper parameters/args from config
    # NOTE: float() is required to parse any exponentials since YAML sends exponentials as strings
    # model_name = config["model_name"]
    # download_base = config["download_base"]
    input_type = config["input_type"]
    trained_checkpoint_prefix = config["trained_checkpoint_prefix"]
    pipeline_config_path = config["pipeline_config_path"]
    output_directory = config["output_directory"]

    # freeze graph
    # Note: for some reason, calling freeze_graph module's freeze_graph from within python doesn't work and complains
    # about lack of all args being supplied, so we have to call it as a terminal command.
    subprocess.call(["python",
                    "/home/nightrider/tensorflow/models/research/object_detection/export_inference_graph.py", # needs absolute path
                    "--input_type", "{}".format(input_type),
                    "--trained_checkpoint_prefix", "{}".format(trained_checkpoint_prefix),
                    "--pipeline_config_path", "{}".format(pipeline_config_path),
                    "--output_directory", "{}".format(output_directory)])