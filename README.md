# calacademy-fish-id
Automated identification of fish species at CalAcademy

# Datasets
[download zip](https://drive.google.com/file/d/1RwhbTESgcAExOoumiftDsnhtuoUcfhv2/view?usp=drive_link)
- create a symlink directory to point to the downloaded folder
  - i.e. `calacademy-fish-id/datasets` symlinks to whereever the downloaded `datasets` folder resides
 
# Outputs
[download zip]()
- create a symlink directory to point to the downloaded folder
  - i.e. `calacademy-fish-id/outputs` symlinks to whereever the downloaded `outputs` folder resides  

# Models
[download zip]()
- The downloaded `classifiers` folder should replace the `calacademy-fish-id/classifiers` directory

## Usage
*Example command for initiating training of object detection model*
- `python tensorflow/models/research/object_detection/model_main.py --pipeline_config_path=/home/nightrider/calacademy-fish-id/classifiers/models/ssd_mobilenet_v2_coco_2018_03_29/pretrained/pipeline.config --model_dir=/home/nightrider/calacademy-fish-id/outputs/fine_tuned_model_training/ssd_mobilenet_v2_coco_2018_03_29/5_7_2020`
