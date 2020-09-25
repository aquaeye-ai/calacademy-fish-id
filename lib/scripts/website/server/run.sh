# Runs the server app

#!/bin/bash

source config.sh

# Expose ports and run
gunicorn3 --bind $HOST model_server_keras_h5:app