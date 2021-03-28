"""
Script that wraps tensorflow's model_main.py functionality so that it can be run as demonstrated here:
https://towardsdatascience.com/detailed-tutorial-build-your-custom-real-time-object-detector-5ade1017fd2d#b948
"""
import os
import yaml
import subprocess

if __name__ == "__main__":
    # we expect, as a hand-shake agreement, that there is a .yml config file in top level of lib/configs directory
    config_dir = os.path.join(os.curdir, '..', '..', 'configs', 'object_detection')
    yaml_path = os.path.join(config_dir, 'finetune_pretrained_tf_model.yml')
    with open(yaml_path, "r") as stream:
        config = yaml.load(stream)

    ## collect hyper parameters/args from config
    # NOTE: float() is required to parse any exponentials since YAML sends exponentials as strings
    # model_name = config["model_name"]
    # download_base = config["download_base"]
    pipeline_config_path = config["pipeline_config_path"]
    model_directory = config["model_directory"]

    subprocess.call(["python",
                    "/home/nightrider/tensorflow/models/research/object_detection/model_main.py", # needs absolute path
                    "--pipeline_config_path", "{}".format(pipeline_config_path),
                    "--model_dir", "{}".format(model_directory)])