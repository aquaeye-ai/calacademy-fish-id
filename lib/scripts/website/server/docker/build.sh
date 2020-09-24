#!/bin/bash

source config.sh

echo "Image name is: ${IMAGE_NAME}"

sudo docker build -t ${IMAGE_NAME} .

