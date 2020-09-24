FROM gcr.io/tensorflow/tensorflow
LABEL maintainer="https://github.com/jdcast/calacademy-fish-id"

#pip install requirements
ADD requirements.txt /
RUN pip install -r /requirements.txt

RUN apt-get update -y

#install gunicorn, needs to be after udpate above
RUN apt-get install -y gunicorn