# calacademy-fish-id
Automated identification of fish species at CalAcademy

## Usage
*Example command for initiating training of object detection model*
- `python tensorflow/models/research/object_detection/model_main.py --pipeline_config_path=/home/nightrider/calacademy-fish-id/classifiers/models/ssd_mobilenet_v2_coco_2018_03_29/pretrained/pipeline.config --model_dir=/home/nightrider/calacademy-fish-id/outputs/fine_tuned_model_training/ssd_mobilenet_v2_coco_2018_03_29/5_7_2020`