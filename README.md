# calacademy-fish-id
Automated identification of fish species at CalAcademy

## Usage
*Example command for training*
- `python tensorflow/models/research/object_detection/model_main.py --pipeline_config_path=/home/nightrider/calacademy-fish-id/classifiers/models/ssd_mobilenet_v2_coco_2018_03_29/pretrained/pipeline.config --model_dir=/home/nightrider/calacademy-fish-id/outputs/fine_tuned_model_training/ssd_mobilenet_v2_coco_2018_03_29/5_7_2020`

*Example command to export graph for inference*
- `python ~/tensorflow/models/research/object_detection/export_inference_graph.py --input_type=image_tensor --trained_checkpoint_prefix=/home/nightrider/calacademy-fish-id/outputs/fine_tuned_model_training/ssd_mobilenet_v2_coco_2018_03_29/5_7_2020/model.ckpt-47309 --pipeline_config_path=/home/nightrider/calacademy-fish-id/classifiers/models/ssd_mobilenet_v2_coco_2018_03_29/pretrained/pipeline.config --output_directory=/home/nightrider/calacademy-fish-id/classifiers/models/ssd_mobilenet_v2_coco_2018_03_29/fine_tuned/5_7_2020`
