"""
Script that wraps tensorflow's freeze_graph functionality so that it can be run as demonstrated here:
https://github.com/tensorflow/models/tree/master/research/slim#Export and
https://stackoverflow.com/questions/51408732/vgg-19-slim-model-a-frozen-pb-graph
"""
import os
import yaml

from tensorflow.python.tools import freeze_graph

if __name__ == "__main__":
    # we expect, as a hand-shake agreement, that there is a .yml config file in top level of lib/configs directory
    config_dir = os.path.join(os.curdir, '..', 'configs')
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
    freeze_graph.freeze_graph(input_graph=input_graph,
                              input_binary=input_binary,
                              input_checkpoint=input_checkpoint,
                              output_node_names=output_node_names,
                              output_graph=output_graph)