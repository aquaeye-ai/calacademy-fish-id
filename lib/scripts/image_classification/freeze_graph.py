"""
Script that wraps tensorflow's freeze_graph functionality so that it can be run as demonstrated here:
https://github.com/tensorflow/models/tree/master/research/slim#Export and
https://stackoverflow.com/questions/51408732/vgg-19-slim-model-a-frozen-pb-graph

Requires /lib/image_classification/export_inference_graph.py's exported_inference_graph.pb output as input
"""
import os
import yaml
import subprocess

# from tensorflow.python.tools import freeze_graph

if __name__ == "__main__":
    # we expect, as a hand-shake agreement, that there is a .yml config file in top level of lib/configs directory
    config_dir = os.path.join(os.curdir, '..', 'configs', 'image_classification')
    yaml_path = os.path.join(config_dir, 'freeze_graph.yml')
    with open(yaml_path, "r") as stream:
        config = yaml.load(stream)

    ## collect hyper parameters/args from config
    # NOTE: float() is required to parse any exponentials since YAML sends exponentials as strings
    # model_name = config["model_name"]
    # download_base = config["download_base"]
    input_graph = config["input_graph"]
    input_checkpoint = config["input_checkpoint"]
    input_binary = True if config["input_binary"] > 0 else False
    output_node_names = config["output_node_names"]
    output_graph = config["output_graph"]

    # freeze graph
    # Note: for some reason, calling freeze_graph module's freeze_graph from within python doesn't work and complains
    # about lack of all args being supplied, so we have to call it as a terminal command.
    subprocess.call(["python",
                    "-m",
                    "tensorflow.python.tools.freeze_graph",
                    "--input_graph", "{}".format(input_graph),
                    "--input_binary", "true",
                    "--output_node_names", "{}".format(output_node_names),
                    "--input_checkpoint", "{}".format(input_checkpoint),
                    "--output_graph", "{}".format(output_graph)])