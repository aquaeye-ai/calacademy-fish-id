#!/bin/bash

source config.sh

read -p "Enter the full path to the github directory.  This will be exposed at /share.  Example: '~/guardian_demo':" PROJ_DIR
PROJ_DIR=$(eval echo $PROJ_DIR)


# Expose ports and run
sudo docker run -it -p $PUBLIC_PORT:$PUBLIC_PORT -v "$PROJ_DIR":/share --name $IMAGE_NAME $IMAGE_NAME
