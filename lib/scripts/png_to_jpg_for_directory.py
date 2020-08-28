"""
Script that wraps mogrify and rm to convert all png images to jpg format and then remove the png images if desired.
"""
import os
import yaml
import subprocess

# from tensorflow.python.tools import freeze_graph

if __name__ == "__main__":
    # we expect, as a hand-shake agreement, that there is a .yml config file in top level of lib/configs directory
    config_dir = os.path.join(os.curdir, 'configs')
    yaml_path = os.path.join(config_dir, 'png_to_jpg_for_directory.yml')
    with open(yaml_path, "r") as stream:
        config = yaml.load(stream)

    ## collect hyper parameters/args from config
    # NOTE: float() is required to parse any exponentials since YAML sends exponentials as strings
    source_directory = config["source_directory"]
    remove_png = config["remove_png"]

    # convert all png to jpg
    mog_process = subprocess.Popen("mogrify -verbose -format jpg {}/*.png".format(source_directory),
                                   shell=True)
                                    # stdout=subprocess.PIPE,
                                    # stderr=subprocess.PIPE)

    # Popen is asynchronous and mogrify takes a while for large directorys, so we must wait for it to complete before
    # kicking off the command to start removing images.  Otherwise, rm will run and remove images before mogrify has
    # converted them.
    mog_process.wait()

    # remove all png if desired
    if remove_png > 0:
        print("Removing png images") # this line seems to prevent a perceived race condition between the mogrify process ending and the rm process beginning -> sometimes the rm process doesn't complete without this line added
        rm_process = subprocess.Popen("rm -v {}/*.png".format(source_directory),
                                      shell=True)
                                        # stdout=subprocess.PIPE,
                                        # stderr=subprocess.PIPE)

        # there is a race condition (I think) where the rm process fails to fully complete, so we force the program to wait on it
        rm_process.wait()