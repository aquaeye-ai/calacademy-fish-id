# Script used to sample video file for frames to annotate.

import os
import yaml

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # we expect, as a hand-shake agreement, that there is a .yml config file in top level of lib/configs directory
    config_dir = os.path.join(os.pardir, 'configs', 'image_classification')
    yaml_path = os.path.join(config_dir, 'pcr_data_stats.yml')
    with open(yaml_path, "r") as stream:
        config = yaml.load(stream)

    ## collect hyper parameters/args from config
    # NOTE: float() is required to parse any exponentials since YAML sends exponentials as strings
    directory = config["directory"]

    ## collect the data for each class

    # get class directories
    class_dirs = [d for d in os.listdir(directory)]

    # collect Google/Bing/Combined subdirectory stats
    num_bing_totals = []
    num_google_totals = []
    num_combined_totals = []
    for class_dir in class_dirs:
        class_dir_path = os.path.join(directory, class_dir)
        bing_dir_path = os.path.join(class_dir_path, 'bing')
        google_dir_path = os.path.join(class_dir_path, 'google')
        combined_dir_path = os.path.join(class_dir_path, 'combined')

        num_bing = 0
        num_google = 0
        num_combined = 0


        if os.path.exists(bing_dir_path): # directory may not exist (i.e. it was removed after combining with google data)
            num_bing = len([f for f in os.listdir(bing_dir_path) if os.path.isfile(os.path.join(bing_dir_path, f))])

        if os.path.exists(google_dir_path): # directory may not exist (i.e. it was removed after combining with bing data)
            num_google = len([f for f in os.listdir(google_dir_path) if os.path.isfile(os.path.join(google_dir_path, f))])

        num_combined = len([f for f in os.listdir(combined_dir_path) if os.path.isfile(os.path.join(combined_dir_path, f))])

        num_bing_totals.append(num_bing)
        num_google_totals.append(num_google)
        num_combined_totals.append(num_combined)

        print("[INFO] In directory {} -> num_bing: {}, num_google: {}, num_combined: {}".format(class_dir, num_bing, num_google, num_combined))

    # sort the data for better viewing
    zipped = zip(num_combined_totals, num_google_totals, num_bing_totals, class_dirs)
    zipped.sort(reverse=True)

    ## perform plotting

    # set width of bar
    barWidth = 0.25

    # Set position of bar on X axis
    r1 = np.arange(len(num_bing_totals))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # Make the plot
    plt.bar(r1, [x[0] for x in zipped], color='#7f6d5f', width=barWidth, edgecolor='white', label='Combined')
    plt.bar(r2, [x[1] for x in zipped], color='#557f2d', width=barWidth, edgecolor='white', label='Google')
    plt.bar(r3, [x[2] for x in zipped], color='#2d7f5e', width=barWidth, edgecolor='white', label='Bing')

    # set yticks
    plt.ylabel('Frequency', fontweight='bold')
    plt.yticks([x + 100 for x in range(0, zipped[0][0], 100)])

    # Add xticks on the middle of the group bars
    plt.xlabel('Class', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(num_bing_totals))], [x[3] for x in zipped])
    plt.xticks(rotation=90)
    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=8)
    ax.yaxis.grid()

    # extend margin at bottom of graph
    plt.tight_layout()

    # Create legend & Show graphic
    plt.legend()
    plt.show()