"""
Script that runs server that serves requests for inference_livestream_keras.py
Should be ran within same directory as script with following command from a terminal: gunicorn model_server_keras_h5:app
"""


import os
import cv2
import yaml
import json
import falcon
import base64
import logging

import numpy as np

from tensorflow import keras


class RequireJSON(object):

    def process_request(self, req, resp):
        if not req.client_accepts_json:
            raise falcon.HTTPNotAcceptable(
                'This API only supports responses encoded as JSON.',
                href='http://docs.examples.com/api/json')

        if req.method in ('POST', 'PUT'):
            if 'application/json' not in req.content_type:
                raise falcon.HTTPUnsupportedMediaType(
                    'This API only supports requests encoded as JSON.',
                    href='http://docs.examples.com/api/json')


class JSONTranslator(object):

    def process_request(self, req, resp):
        # req.stream corresponds to the WSGI wsgi.input environ variable,
        # and allows you to read bytes from the request body.
        #
        # See also: PEP 3333
        if req.content_length in (None, 0):
            # Nothing to do
            return

        body = req.stream.read()
        if not body:
            raise falcon.HTTPBadRequest('Empty request body',
                                        'A valid JSON document is required.')

        try:
            req.context['doc'] = json.loads(body.decode('utf-8'))

        except (ValueError, UnicodeDecodeError):
            raise falcon.HTTPError(falcon.HTTP_753,
                                   'Malformed JSON',
                                   'Could not decode the request body. The '
                                   'JSON was incorrect or not encoded as '
                                   'UTF-8.')

    def process_response(self, req, resp, resource, req_succeeded):
        if not hasattr(resp.context, 'result'):
            return

        resp.body = json.dumps(resp.context.result)


# Sample resource from the docs to test the api is running correctly.
class QuoteResource:
    def on_get(self, req, resp):
        """Handles GET requests"""
        quote = {
            'quote': 'I\'ve always been more interested in the future than in the past.',
            'author': 'Grace Hopper'
        }

        resp.body = json.dumps(quote)


# Resource to provide the number of classes for input scales/sliders on FE gui.
class NumClassesResource:
    def init(self, num_classes=None):
        self.num_classes = num_classes

    def on_get(self, req, resp):
        """Handles GET requests"""
        response = {}
        response['num_classes'] = self.num_classes

        resp.body = json.dumps(response)


# Manage evaluation requests.
class EvalResource(object):

    def __init__(self):
        self.logger = logging.getLogger('modelserver.' +  __name__)

    def init(self, model=None, model_input_height=None, model_input_width=None, scale=None, category_index=None):
        self.model = model
        self.model_input_height = model_input_height
        self.model_input_width = model_input_width
        self.scale = scale
        self.category_index = category_index

    def on_post(self, req, resp):

        try:
            # print(req)
            doc = req.context.doc
        # print(doc)
        except AttributeError:
            raise falcon.HTTPBadRequest(
                'Missing thing',
                'A thing must be submitted in the request body.')

        doc_str = json.loads(doc)

        # Get the image id, height, width and data from the JOSN msg.
        image = base64.b64decode(doc_str['image'])
        image = np.frombuffer(image, dtype=np.uint8).copy()
        image = cv2.imdecode(image, flags=1)
        id = doc_str['id']
        height = doc_str['height']
        width = doc_str['width']
        depth = doc_str['depth']
        K = doc_str['K']

        print("id: {}".format(id))
        print("source shape: {}x{}".format(height, width))
        print("target shape: {}x{}".format(self.model_input_height, self.model_input_width))

        # scale image since we trained on scaled images
        image = image * (1. / self.scale)

        # reshape into original dimensions
        image = np.reshape(image, (height, width, depth))

        # resize image to fit model input dimensions
        image = cv2.resize(image, dsize=(self.model_input_height, self.model_input_width), interpolation=cv2.INTER_LINEAR)
        image = np.expand_dims(image, axis=0)

        # Perform evaluation of the image
        output_dict = self.model.predict(image)

        class_scores = output_dict[0]

        # sort the class_scores
        top_k_scores_idx = np.argsort(class_scores)[-K:]
        top_k_scores_idx = list(reversed(top_k_scores_idx))
        top_k_scores = class_scores[top_k_scores_idx]

        ## return the top-k classes and scores to text file
        class_labels = [category_index[i] for i in top_k_scores_idx]

        # Create the response message
        response = {}
        response['id'] = id
        response['top_k_classes'] = class_labels
        response['top_k_scores'] = [str(i) for i in top_k_scores]

        # Return the response message
        resp.body = json.dumps(response)
        resp.status = falcon.HTTP_201


# this can't be in __main__ for some reason...TODO find out why
app = falcon.API(middleware=[
        RequireJSON(),
        JSONTranslator(),
    ])
eval_resource = EvalResource()
num_classes_resource = NumClassesResource()
app.add_route('/eval', eval_resource)
app.add_route('/num_classes', num_classes_resource)
app.add_route('/quote', QuoteResource())


if __name__ == 'model_server_keras_h5':
    # we expect, as a hand-shake agreement, that there is a .yml config file in top level of lib/configs directory
    config_dir = os.path.join('.')
    yaml_path = os.path.join(config_dir, 'model_server_keras_h5.yml')
    with open(yaml_path, "r") as stream:
        config = yaml.load(stream)

    ## collect hyper parameters/args from config
    # NOTE: float() is required to parse any exponentials since YAML sends exponentials as strings
    # model_name = config["model_name"]
    # download_base = config["download_base"]
    path_to_model_h5 = config["model"]
    path_to_labels = config["labels"]
    print(os.listdir('.'))
    model_input_height = config["model_input_height"]
    model_input_width = config["model_input_width"]
    scale = config["scale"]

    # Dictionary of the strings that is used to add correct label for each class index in the model's output.
    # key: index in output
    # value: string name of class
    category_index = {}
    with open(path_to_labels) as labels_f:
        # we need to sort the labels since this is how keras reads the labels in during training: https://stackoverflow.com/questions/38971293/get-class-labels-from-keras-functional-model
        idx = 0
        for line in sorted(labels_f):
            category_index[idx] = line.strip()
            idx += 1

    num_classes_resource.init(num_classes=len(category_index.keys()))

    print("Loading model server.")
    model = keras.models.load_model(path_to_model_h5)
    eval_resource.init(model=model,
                       model_input_height=model_input_height,
                       model_input_width=model_input_width,
                       scale=scale,
                       category_index=category_index)

    # # Expose port, transition to gunicorn or similar if needed.
    # print("Making simple server.")
    # httpd = simple_server.make_server('127.0.0.1', 8000, app)
    # print("Simple server made, starting server.")
    # httpd.serve_forever()
    # print("Server closed.")