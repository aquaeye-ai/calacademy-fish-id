#!/bin/bash

source config.sh

read -p "Enter the full path to the github directory.  This will be exposed at /share.  Example: '~/aquaeye_ai':" PROJ_DIR
PROJ_DIR=$(eval echo $PROJ_DIR)

read -p "Enter the full path to the model files directory.  This will be exposed at /share/model.  Example: '~/aquaeye_ai/models':" MODEL_DIR
MODEL_DIR=$(eval echo $MODEL_DIR)

# Expose ports and run
sudo docker run -it \
        -p $PUBLIC_PORT:$PUBLIC_PORT \
        -v "$PROJ_DIR":/share \
        -v "$MODEL_DIR":/share/model \
        --name $IMAGE_NAME $IMAGE_NAME