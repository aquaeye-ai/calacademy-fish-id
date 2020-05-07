# by Chengwei
import os
import re
import numpy as np


training_dir = '/home/nightrider/calacademy-fish-id/outputs/fine_tuned_model_training/ssd_mobilenet_v2_coco_2018_03_29/5_4_2020'

lst = os.listdir(training_dir)
lst = [l for l in lst if 'model.ckpt-' in l and '.meta' in l]
steps=np.array([int(re.findall('\d+', l)[0]) for l in lst])
last_model = lst[steps.argmax()].replace('.meta', '')

last_model_path = os.path.join(training_dir, last_model)

print(last_model_path)