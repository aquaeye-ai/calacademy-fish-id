import os
import yaml

import pandas as pd
import matplotlib.pyplot as plt

TRAIN = "train"
TEST = "test"

def add_labels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha='center')

def generate_stats(directory=None, db_type=None):
    ## read appropriate csv file and collect the data for each class

    csv_file_path = os.path.join(directory, "{}_labels.csv".format(db_type))
    df = pd.read_csv(csv_file_path, usecols=["class"])

    # save dict of scene classes/frequencies
    class_dict = {}

    for idx in df.index:
        # print(idx)
        class_name = df.values[idx][0]

        if class_name in class_dict:
            class_dict[class_name] += 1
        else:
            class_dict[class_name] = 0

    print(class_dict)

    # sort dict for better viewing
    class_dict = {k: v for k, v in sorted(class_dict.items(), key=lambda item: item[1])}
    class_sorted = sorted(class_dict.items(), key=lambda x:x[1], reverse=True)
    x = []
    y = []
    for cls_tup in class_sorted:
        x.append(cls_tup[0])
        y.append(cls_tup[1])

    plt.bar(range(len(x)), list(y), align='center')
    add_labels(x, y)
    plt.xticks(range(len(x)), x)
    plt.xticks(rotation=90)

    ax = plt.gca()
    # ax.tick_params(axis='x', labelsize=8)
    ax.yaxis.grid()

    # extend margin at bottom of graph
    plt.tight_layout()
    plt.savefig(os.path.join(directory, '{}_stats.png'.format(db_type)), bbox_inches='tight', pad_inches=0.0)

    plt.show()


if __name__ == "__main__":
    # we expect, as a hand-shake agreement, that there is a .yml config file in top level of lib/configs directory
    config_dir = os.path.join('..', '..', 'configs',  'object_detection')
    yaml_path = os.path.join(config_dir, 'data_stats.yml')
    with open(yaml_path, "r") as stream:
        config = yaml.load(stream)

    ## collect hyper parameters/args from config
    # NOTE: float() is required to parse any exponentials since YAML sends exponentials as strings
    directory = config["directory"]
    db_type = config["db_type"]

    if db_type == TRAIN or db_type == TEST:
        generate_stats(directory=directory, db_type=db_type)
