"""
Script that wraps tensorflow/models/research/slim/export_inference_graph.py functionality so that it can be run as demonstrated here:
https://github.com/tensorflow/models/tree/master/research/slim#Export

Provides the necessary exported_inference_graph.pb that is used as input to /lib/image_classification/freeze_graph.py
"""
import os
import yaml
import subprocess


if __name__ == "__main__":
    # we expect, as a hand-shake agreement, that there is a .yml config file in top level of lib/configs directory
    config_dir = os.path.join(os.curdir, '..', 'configs', 'image_classification')
    yaml_path = os.path.join(config_dir, 'export_inference_graph.yml')
    with open(yaml_path, "r") as stream:
        config = yaml.load(stream)

    ## collect hyper parameters/args from config
    # NOTE: float() is required to parse any exponentials since YAML sends exponentials as strings
    # model_name = config["model_name"]
    # download_base = config["download_base"]
    model_name = config["model_name"]
    output_file = config["output_file"]

    # freeze graph
    # Note: for some reason, calling freeze_graph module's freeze_graph from within python doesn't work and complains
    # about lack of all args being supplied, so we have to call it as a terminal command.
    subprocess.call(["python",
                    "/home/nightrider/tensorflow/models/research/slim/export_inference_graph.py",
                    "--alsologtostderr",
                    "--model_name", "{}".format(model_name),
                    "--output_file", "{}".format(output_file)])