# Script used to generate chart graphics representing data statistics for each dataset within dataset/image_classification/pcr.
# NOTE: will extract the dataset name from the dataset path.  Thus, the basename of a given dataset path must match one
# of the predefined names at the start of this file, e.g. 'master' or 'scientific_species_names'.

import os
import yaml

import numpy as np
import matplotlib.pyplot as plt

MASTER_DB = "master"
SCIENTIFIC_SPECIES_NAMES_DB = "scientific_species_names"
COMMON_GROUP_NAMES_DB = "common_group_names"
TRAINING_SPLITS_DB = "training_splits"


def generate_master_stats(directory=None):
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

        if os.path.exists(
                bing_dir_path):  # directory may not exist (i.e. it was removed after combining with google data)
            num_bing = len([f for f in os.listdir(bing_dir_path) if os.path.isfile(os.path.join(bing_dir_path, f))])

        if os.path.exists(
                google_dir_path):  # directory may not exist (i.e. it was removed after combining with bing data)
            num_google = len(
                [f for f in os.listdir(google_dir_path) if os.path.isfile(os.path.join(google_dir_path, f))])

        num_combined = len(
            [f for f in os.listdir(combined_dir_path) if os.path.isfile(os.path.join(combined_dir_path, f))])

        num_bing_totals.append(num_bing)
        num_google_totals.append(num_google)
        num_combined_totals.append(num_combined)

        print("[INFO] In directory {} -> num_bing: {}, num_google: {}, num_combined: {}".format(class_dir, num_bing,
                                                                                                num_google,
                                                                                                num_combined))

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

    # Create legend, title & Show graphic
    plt.title("master dataset stats")
    plt.legend()
    plt.show()

def generate_common_group_names_stats(directory=None):
    ## collect the data for each class

    # get class directories
    class_dirs = [d for d in os.listdir(directory)]

    # collect Google/Bing/Combined subdirectory stats
    num_totals = []
    for class_dir in class_dirs:
        class_dir_path = os.path.join(directory, class_dir)

        num = len([f for f in os.listdir(class_dir_path) if os.path.isfile(os.path.join(class_dir_path, f))])

        num_totals.append(num)

        print("[INFO] In directory {} -> num: {}".format(class_dir, num))

    # sort the data for better viewing
    zipped = zip(num_totals, class_dirs)
    zipped.sort(reverse=True)

    ## perform plotting

    # set width of bar
    barWidth = 0.25

    # Set position of bar on X axis
    r1 = np.arange(len(num_totals))

    # Make the plot
    plt.bar(r1, [x[0] for x in zipped], color='#7f6d5f', width=barWidth, edgecolor='white', label='Total')

    # set yticks
    plt.ylabel('Frequency', fontweight='bold')
    plt.yticks([x + 250 for x in range(0, zipped[0][0], 250)])

    # Add xticks on the middle of the group bars
    plt.xlabel('Class', fontweight='bold')
    plt.xticks([r for r in range(len(num_totals))], [x[1] for x in zipped])
    plt.xticks(rotation=90)
    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=8)
    ax.yaxis.grid()

    # extend margin at bottom of graph
    plt.tight_layout()

    # Create legend, title & Show graphic
    plt.title("common_group_names dataset stats")
    plt.legend()
    plt.show()

def generate_scientific_species_names_stats(directory=None):
    ## collect the data for each class

    # get class directories
    class_dirs = [d for d in os.listdir(directory)]

    # collect Google/Bing/Combined subdirectory stats
    num_totals = []
    for class_dir in class_dirs:
        class_dir_path = os.path.join(directory, class_dir)

        num = len([f for f in os.listdir(class_dir_path) if os.path.isfile(os.path.join(class_dir_path, f))])

        num_totals.append(num)

        print("[INFO] In directory {} -> num: {}".format(class_dir, num))

    # sort the data for better viewing
    zipped = zip(num_totals, class_dirs)
    zipped.sort(reverse=True)

    ## perform plotting

    # set width of bar
    barWidth = 0.25

    # Set position of bar on X axis
    r1 = np.arange(len(num_totals))

    # Make the plot
    plt.bar(r1, [x[0] for x in zipped], color='#7f6d5f', width=barWidth, edgecolor='white', label='Total')

    # set yticks
    plt.ylabel('Frequency', fontweight='bold')
    plt.yticks([x + 250 for x in range(0, zipped[0][0], 250)])

    # Add xticks on the middle of the group bars
    plt.xlabel('Class', fontweight='bold')
    plt.xticks([r for r in range(len(num_totals))], [x[1] for x in zipped])
    plt.xticks(rotation=90)
    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=8)
    ax.yaxis.grid()

    # extend margin at bottom of graph
    plt.tight_layout()

    # Create legend, title & Show graphic
    plt.title("scientific_species_names dataset stats")
    plt.legend()
    plt.show()

def generate_training_splits_stats(directory=None):
    generate_training_split_stats(directory=os.path.join(directory, 'train'), split='train')
    generate_training_split_stats(directory=os.path.join(directory, 'val'), split='val')
    generate_training_split_stats(directory=os.path.join(directory, 'test'), split='test')

def generate_training_split_stats(directory=None, split=None):
    ## collect the data for each class

    # get class directories
    class_dirs = [d for d in os.listdir(directory)]

    # collect Google/Bing/Combined subdirectory stats
    num_web_totals = []
    num_od_totals = []
    num_combined_totals = []
    for class_dir in class_dirs:
        class_dir_path = os.path.join(directory, class_dir)

        num_web = len([f for f in os.listdir(class_dir_path) if os.path.isfile(os.path.join(class_dir_path, f))
                       and '_web.' in os.path.join(class_dir_path, f)])

        num_od = len([f for f in os.listdir(class_dir_path) if os.path.isfile(os.path.join(class_dir_path, f))
                       and '_od.' in os.path.join(class_dir_path, f)])

        num_combined = len(
            [f for f in os.listdir(class_dir_path) if os.path.isfile(os.path.join(class_dir_path, f))])

        num_web_totals.append(num_web)
        num_od_totals.append(num_od)
        num_combined_totals.append(num_combined)

        print("[INFO] In directory {} -> num_web: {}, num_od: {}, num_combined: {}".format(class_dir, num_web, num_od,
                                                                                           num_combined))

    # sort the data for better viewing
    zipped = zip(num_combined_totals, num_web_totals, num_od_totals, class_dirs)
    zipped.sort(reverse=True)

    ## perform bar plotting

    # set width of bar
    barWidth = 0.25

    # Set position of bar on X axis
    r1 = np.arange(len(num_web_totals))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # Make the plot
    plt.bar(r1, [x[0] for x in zipped], color='#7f6d5f', width=barWidth, edgecolor='white', label='Combined')
    plt.bar(r2, [x[1] for x in zipped], color='#557f2d', width=barWidth, edgecolor='white', label='Web')
    plt.bar(r3, [x[2] for x in zipped], color='#2d7f5e', width=barWidth, edgecolor='white', label='Object Detection')

    # set yticks
    plt.ylabel('Frequency', fontweight='bold')
    plt.yticks([x + 100 for x in range(0, zipped[0][0], 100)])

    # Add xticks on the middle of the group bars
    plt.xlabel('Class', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(num_web_totals))], [x[3] for x in zipped])
    plt.xticks(rotation=90)
    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=8)
    ax.yaxis.grid()

    # extend margin at bottom of graph
    plt.tight_layout()

    # Create legend, title & Show graphic
    plt.title("Training Split Stats: {}".format(split))
    plt.legend()
    # plt.savefig(os.path.join(os.path.dirname(directory), 'training_split_{}_stats.png'.format(split)))
    plt.show()
    plt.close()

    ## perform pi charting

    labels = ['Web', 'Object Detection']
    sizes = [np.float(sum(num_web_totals)) / np.float(sum(num_combined_totals)), np.float(sum(num_od_totals)) / np.float(sum(num_combined_totals))]
    explode = (0, 0)
    fig2, ax2 = plt.subplots()
    ax2.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=False, startangle=180)
    ax2.axis('equal')
    plt.title("{}: Web Vs Object Detection".format(split))
    plt.show()


if __name__ == "__main__":
    # we expect, as a hand-shake agreement, that there is a .yml config file in top level of lib/configs directory
    config_dir = os.path.join(os.pardir, 'configs', 'image_classification')
    yaml_path = os.path.join(config_dir, 'pcr_data_stats.yml')
    with open(yaml_path, "r") as stream:
        config = yaml.load(stream)

    ## collect hyper parameters/args from config
    # NOTE: float() is required to parse any exponentials since YAML sends exponentials as strings
    directory = config["directory"]
    db_type = config["db_type"]

    if db_type == MASTER_DB:
        generate_master_stats(directory=directory)
    elif db_type == COMMON_GROUP_NAMES_DB:
        generate_common_group_names_stats(directory=directory)
    elif db_type == SCIENTIFIC_SPECIES_NAMES_DB:
        generate_scientific_species_names_stats(directory=directory)
    elif db_type == TRAINING_SPLITS_DB:
        generate_training_splits_stats(directory=directory)

