import numpy as np
import os
import cv2
import yaml
import Tkinter as tk
import Image, ImageTk


# if __name__ == "__main__":
#     # we expect, as a hand-shake agreement, that there is a .yml config file in top level of lib/configs directory
#     config_dir = os.path.join(os.pardir, 'configs', 'image_classification')
#     yaml_path = os.path.join(config_dir, 'inference_keras_model_on_livestream.yml')
#     with open(yaml_path, "r") as stream:
#         config = yaml.load(stream)
#
#     ## collect hyper parameters/args from config
#     # NOTE: float() is required to parse any exponentials since YAML sends exponentials as strings
#     stream_url = config["stream_url"]
#     dst_dir = config["destination_directory"]
#
#     ## set up GUI
#
#     # window
#     window = tk.Tk()  #Makes main window
#     window.title("Reef Lagoon")
#     window.config(background="#FFFFFF")
#     window.bind('<Escape>', lambda e: window.quit())
#
#     # graphics frame
#     gf = tk.Frame(window, width=600, height=500)
#     gf.grid(row=0, column=0, padx=10, pady=2)
#
#     # controls frame
#     cf = tk.Frame(window, width=600, height=100)
#     cf.grid(row=100, column=0, padx=10, pady=2)
#
#     # capture video frames
#     lmain = tk.Label(gf)
#     lmain.grid(row=0, column=0)
#     # cap = cv2.VideoCapture(0)
#     cap =  cv2.VideoCapture(stream_url)
#     def show_frame():
#         _, frame = cap.read()
#         frame = cv2.flip(frame, 1)
#         cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
#         cv2image = cv2.resize(cv2image, dsize=(600,500))
#         img = Image.fromarray(cv2image)
#         imgtk = ImageTk.PhotoImage(image=img)
#         lmain.imgtk = imgtk
#         lmain.configure(image=imgtk)
#         lmain.after(10, show_frame)
#
#     show_frame()  #Display 2
#     window.mainloop()  #Starts GUI

import numpy as np
import cv2
import os

import json, simplejson
import requests
import time

import base64

import Tkinter as tk
from PIL import Image, ImageTk
import datetime

from collections import Counter

# SERVER_URL = 'http://172.17.0.2:8000/eval'


class webcam_manager():
    def __init__(self, stream_url=None, server_url=None, server_eval_endpoint=None, server_num_classes_endpoint=None,
                 num_classes=None):
        # init state
        self.shouldPause = False
        self.shouldHideModelControls = False
        self.frame_count = 0
        self.frame = None
        self.stream_url = stream_url
        self.server_url = server_url
        self.server_eval_endpoint = server_eval_endpoint
        self.server_num_classes_endpoint = server_num_classes_endpoint
        self.font = font = cv2.FONT_HERSHEY_SIMPLEX
        self.width, self.height = 800, 600
        self.classification_buffer = []
        self.classification_buffer_index = 0
        self.classification_buffer_ready = False
        self.unsure_threshold = 0.5
        self.K = 1
        self.num_classes = num_classes
        self.drone_db = {'phantom': {'freqs': ['5.725 GHz - 5.825 GHz', '922.7 MHz - 927.7 MHz'],
                                     'maxTransmit': '1000m',
                                     'maxSpeed': '16m/s',
                                     'maxFlightTime': '25min',
                                     'maxPayloadWeight': '300g',
                                     'vulnerabilities': 'GPS Spoofing'},
                         'parrot': {'freqs': ['2.4 GHz'],
                                    'maxTransmit': '100m',
                                    'maxSpeed': '11.1m/s',
                                    'maxFlightTime': '12min',
                                    'maxPayloadWeight': '100g',
                                    'vulnerabilities': 'Open Wifi Communication'}}

        # init UI elements
        self.init_cap()
        self.init_tk_root()
        self.init_tk_frames()
        self.init_tk_labels()
        self.init_tk_buttons()
        self.init_tk_scales()

    def init_cap(self):
        self.cap = cv2.VideoCapture(self.stream_url)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def init_tk_root(self):
        self.root = tk.Tk()
        self.root.title("Reef Lagoon")
        self.root.configure(background='gray')
        self.root.bind('<Escape>', lambda e: self.root.quit())

    def init_tk_frames(self):
        self.init_tk_lFrame()
        self.init_tk_rFrame()
        self.init_tk_inputsFrame()
        self.init_tk_cmFrame()
        self.init_tk_pcFrame()
        self.init_tk_mcFrame()

    def init_tk_labels(self):
        self.init_tk_imageLabel()
        self.init_tk_inputsLabel()
        self.init_tk_cmLabel()

    def init_tk_buttons(self):
        self.init_tk_jammingButton()
        self.init_tk_empButton()
        self.init_tk_hackButton()
        self.init_tk_startButton()
        self.init_tk_quitButton()
        self.init_tk_pauseButton()
        self.init_tk_evalButton()
        self.init_tk_hideShowMCButton()

    def init_tk_scales(self):
        self.init_tk_unsure_thrshld_scale()
        self.init_k_scale()

    def init_tk_lFrame(self):
        self.lFrame = tk.LabelFrame(self.root, relief="ridge", borderwidth=4, background="red", text="Frame",
                                    font="bold", labelanchor="n")
        self.lFrame.grid(row=0, column=0, padx=10, pady=10, rowspan=4)

    def init_tk_rFrame(self):
        self.rFrame = tk.Frame(self.root)
        self.rFrame.grid(row=0, column=1, padx=10, pady=10)

    def init_tk_inputsFrame(self):
        self.inputsFrame = tk.LabelFrame(self.root, text="Input", font="bold", labelanchor="n")
        self.inputsFrame.grid(row=0, column=1, padx=10, pady=10)

    def init_tk_cmFrame(self):
        self.cmFrame = tk.LabelFrame(self.root, text="Recommmended Counter Measure", font="bold", labelanchor="n")
        self.cmFrame.grid(row=1, column=1, padx=10, pady=10)

    def init_tk_pcFrame(self):
        self.pcFrame = tk.LabelFrame(self.root, text="Program Controls", font="bold", labelanchor="n")
        self.pcFrame.grid(row=2, column=1, padx=10, pady=10)

    def init_tk_mcFrame(self):
        self.mcFrame = tk.LabelFrame(self.root, text="Model Controls", font="bold", labelanchor="n")
        self.mcFrame.grid(row=3, column=1, padx=10, pady=10)

    def init_tk_imageLabel(self):
        self.imageLabel = tk.Label(self.lFrame, relief="sunken")
        self.imageLabel.grid(row=0, column=0, padx=2, pady=2)

    def init_tk_inputsLabel(self):
        self.inputsLabel = tk.Label(self.inputsFrame, anchor="w", width=75, justify="left", relief="ridge", borderwidth=2)
        self.inputsLabel.grid(row=0, column=0, padx=10, pady=10, columnspan=3)

    def init_tk_cmLabel(self):
        self.cmLabel = tk.Label(self.cmFrame, anchor="w", width=75, justify="left", relief="ridge", borderwidth=2)
        self.cmLabel.grid(row=0, column=0, padx=10, pady=10, columnspan=3)

    def init_tk_jammingButton(self):
        self.jammingButton = tk.Button(self.cmFrame, width=20, relief="raised", borderwidth=2, text="Jam")
        self.jammingButton.grid(row=1, column=0, padx=10, pady=10)

    def init_tk_empButton(self):
        self.empButton = tk.Button(self.cmFrame, width=20, relief="raised", borderwidth=2, text="EMP")
        self.empButton.grid(row=1, column=1, padx=5, pady=10)

    def init_tk_hackButton(self):
        self.hackButton = tk.Button(self.cmFrame, width=20, relief="raised", borderwidth=2, text="Hack")
        self.hackButton.grid(row=1, column=2, padx=10, pady=10)

    def init_tk_startButton(self):
        self.startButton = tk.Button(self.pcFrame, width=13, relief="raised", borderwidth=2, text="Start",
                                     command=self.start)
        self.startButton.grid(row=1, column=0, padx=10, pady=10)

    def init_tk_quitButton(self):
        self.quitButton = tk.Button(self.pcFrame, width=13, relief="raised", borderwidth=2, text="Quit",
                                    command=self.quit)
        self.quitButton.grid(row=1, column=1, padx=10, pady=10)

    def init_tk_pauseButton(self):
        self.pauseButton = tk.Button(self.pcFrame, width=13, relief="raised", borderwidth=2, text="Pause",
                                     command=self.pause)
        self.pauseButton.grid(row=1, column=2, padx=10, pady=10)

    def init_tk_evalButton(self):
        self.evalButton = tk.Button(self.pcFrame, width=13, relief="raised", borderwidth=2, text="Predict!",
                                    command=self.eval)
        self.evalButton.grid(row=2, column=0, padx=10, pady=10)

    def init_tk_hideShowMCButton(self):
        self.hideMCButton = tk.Button(self.pcFrame, width=13, relief="raised", borderwidth=2,
                                      text="Hide/Show Model Controls", wraplength=90,
                                      command=self.hideShowModelControls)
        self.hideMCButton.grid(row=1, column=3, padx=10, pady=10)

    def init_tk_unsure_thrshld_scale(self):
        self.utScale = tk.Scale(self.mcFrame, from_=0, to=1, resolution=0.01, label="UNSURE_THRESHOLD",
                                command=self.utScaleChanged)
        self.utScale.grid(row=0, column=0, padx=10, pady=10)
        self.utScale.set(self.unsure_threshold)

    def init_k_scale(self):
        self.kScale = tk.Scale(self.mcFrame, from_=1, to=self.num_classes, resolution=1, label="K: How Predictions to Return?", command=self.kScaleChanged)
        self.kScale.grid(row=0, column=1, padx=10, pady=10)
        self.kScale.set(self.K)

    def utScaleChanged(self, event):
        self.unsure_threshold = self.utScale.get()

    def kScaleChanged(self, event):
        self.K = self.kScale.get()

    def hideShowModelControls(self):
        self.shouldHideModelControls = ~self.shouldHideModelControls
        if self.shouldHideModelControls:
            self.mcFrame.grid_remove()
        else:
            self.mcFrame.grid()

    def display_cap(self):
        """
        Display frames from the camera until q is pressed.
        :return: None
        """
        # Capture frame-by-frame
        ret, frame = self.cap.read()

        # Display the resulting frame
        if not self.shouldPause:
            frame = cv2.flip(frame, 1)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            cv2image = cv2.resize(cv2image, dsize=(self.width, self.height))

            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)

            self.imageLabel.imgtk = imgtk
            self.imageLabel.configure(image=imgtk)

            self.imageLabel.after(1, self.display_cap)

            self.frame_count += 1

    def write_cap(self, frame_count=300):
        """
         Record a stream of frames (no evaluation)
        :param frame_count: number of frames to record
        :return: None
        """
        for i in xrange(0, frame_count):
            ret, frame = self.cap.read()

            frame_counter = str(i).zfill(len(str(frame_count)))

            cv2.imwrite('capture_tmp/cam_{}_img_{}.jpg'.format(self.camera_id, frame_counter), frame)

            cv2.putText(frame, 'Recording to Disk {}'.format(frame_counter), (10, 50), self.font, 1, (255, 255, 255), 2)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def replay_cap(self):
        image_list = sorted(os.listdir('capture_tmp/'))
        print(image_list)
        image_list = [image_path for image_path in image_list if image_path[-4:] == '.jpg']

        for image_path in image_list:
            image = cv2.imread('capture_tmp/{}'.format(image_path))
            cv2.imshow('frame', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def species_specs(self, drone=""):
        """
        Return species specs for species in text block with each line corresponding to key spec
        :param drone: species db key
        :return: None
        """
        specs = self.drone_db[drone]
        ret = ""
        ret += "Communication Frequencies: {}\n".format(specs['freqs'])
        ret += "Max Controller Distance: {}\n".format(specs['maxTransmit'])
        ret += "Max Speed: {}\n".format(specs['maxSpeed'])
        ret += "Max Flight Time: {}\n".format(specs['maxFlightTime'])
        ret += "Max Payload Weight: {}\n".format(specs['maxPayloadWeight'])
        ret += "Known Vulnerabilities: {}\n".format(specs['vulnerabilities'])
        return ret

    def eval_cap(self):
        """
        Run and evaluate a stream of frames from the camera.
        :return: None
        """
        if self.shouldEval:
            frame = self.frame
            print("image shape: {}".format(frame.shape))
            print("image dtype: {}".format(frame.dtype))

            retval, buffer = cv2.imencode('.jpg', frame)

            response = self.eval_image(id=self.frame_count, shape=frame.shape, data=buffer)

            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            frame = cv2.flip(frame, 1)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            cv2image = cv2.resize(cv2image, dsize=(self.width, self.height))

            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)

            self.imageLabel.imgtk = imgtk
            self.imageLabel.configure(image=imgtk)

            inputsLabelText = "Timestamp: {}\n".format(st)

            # write top K classes and scores
            inputsLabelText += "\nTop {} Classes:\n".format(self.K)
            for idx, _class in enumerate(response['top_k_classes']):
                inputsLabelText += "{}: {}\n".format(_class, response['top_k_scores'][idx])

            # write all classes and their average scores
            inputsLabelText += "\nAverage Scores for All Classes:\n"
            for idx, average_class in enumerate(response['average_classes']):
                inputsLabelText += "{}: {}\n".format(average_class, response['average_scores'][idx])

            self.inputsLabel.text = inputsLabelText
            self.inputsLabel.configure(text=inputsLabelText)

            # if response.json()['evaluation'] == "phantom no payload":
            # if response['payload'] == False:
            #     cmLabelText = "No payload -> Risk level low -> wait and observe"
            # elif response['payload_type'] == 'camera':
            #     # elif response.json()['evaluation'] == "phantom with camera":
            #     cmLabelText = "Camera -> Risk level medium -> Jamming, Hacking"
            # # else:
            # elif response['payload_type'] == 'unknown package':
            #     cmLabelText = "Unidentified payload -> Risk level high -> Jamming, EMP, Hacking"
            # else:
            #     cmLabelText = "User input needed: {}".format(response['payload_type'])
            # self.cmLabel.text = cmLabelText
            # self.cmLabel.configure(text=cmLabelText)
            # self.cmLabel.after(1, self.eval_cap)

    def eval_image(self, id=None, shape=None, data=None):
        """
        Evaluate a single frame. Sends data to model server api and retrieves response.
        TODO: Currently runs about 0.05 seconds round trip. Look for where this could be sped up.
        :param id: id of image
        :param shape: shape of original image
        :param data: image pixel values
        :return: parsed prediction response from server for image
        """
        json_payload = json.dumps({'id': id,
                                   'height': shape[0],
                                   'width': shape[1],
                                   'depth': shape[2],
                                   'image': base64.b64encode(data),
                                   'K': self.K})

        print("Sending request - image id: {}".format(id))
        start = time.time()
        response = requests.post(self.server_url + self.server_eval_endpoint, json=json_payload)
        print('request returned {} in {} seconds.'.format(response.status_code, time.time() - start))
        print('response:\n{}'.format(response.json()))

        parsed_response = self.parse_eval_response(response.json(), id)
        print("parsed response:\n{}".format(parsed_response))
        return parsed_response

    def parse_eval_response(self, response, id):
        """
        Takes the json returned from the model api for a evaluated frame and returns classification broken down by drone,
        payload with confidence, thresholding and smoothing over X frames.
        :param response: json response to parse
        :param id: id of image
        :return: dict containing parsed json response
        """
        # Put threshold on invdividual classifications, if not over the mark, put Unsure, best guess: {} with {}% confidence
        UNSURE_THRESHOLD = self.unsure_threshold

        # number of frame to report average scores
        CLASSIFICATION_BUFFER_LENGTH = 10

        ## parse the response to get classes and scores.

        # convert unicode values to str
        top_k_classes = [str(i) for i in response['top_k_classes']]

        # convert unicode values to float
        top_k_scores = response['top_k_scores']
        top_k_scores = [float(i) for i in top_k_scores]

        # Turn classes/scores list into dict of class: score
        class_scores_dict = {_class: top_k_scores[index] for index, _class in enumerate(top_k_classes)}

        ## compute average class scores

        # for first ten, don't smooth, just fill buffer, otherwise update the buffer
        if self.classification_buffer_ready:
            self.classification_buffer[self.classification_buffer_index] = class_scores_dict
            self.classification_buffer_index += 1
        else:
            self.classification_buffer.append(class_scores_dict)
            self.classification_buffer_index += 1
            if self.classification_buffer_index >= CLASSIFICATION_BUFFER_LENGTH:
                self.classification_buffer_ready = True

        if self.classification_buffer_index >= CLASSIFICATION_BUFFER_LENGTH:
            self.classification_buffer_index = 0

        # compute the average
        # initialize the averages to current values in case the buffer isn't ready
        average_scores_dict = class_scores_dict
        if self.classification_buffer_ready:
            counter_sum = sum((Counter(dict(x)) for x in self.classification_buffer), Counter())
            average_scores_dict = {key: value / (1.0 * CLASSIFICATION_BUFFER_LENGTH) for key, value in counter_sum.iteritems()}

        average_classes = []
        average_scores = []
        for key, value in average_scores_dict.iteritems():
            average_classes.append(key)
            average_scores.append(value)

        # Create response dict:
        parsed_response = {'top_k_classes': top_k_classes,
                           'top_k_scores': top_k_scores,
                           'average_classes': average_classes,
                           'average_scores': average_scores}

        return parsed_response

    def run(self):
        self.display_cap()

    def start(self):
        self.shouldPause = False
        self.run()
        self.startButton.configure(state="disabled")
        self.evalButton.configure(state="disabled")
        self.pauseButton.configure(state="normal")

    def quit(self):
        self.root.quit()

    def pause(self):
        self.shouldPause = True
        self.pauseButton.configure(state="disabled")
        self.startButton.configure(state="normal")
        self.evalButton.configure(state="normal")
        ret, frame = self.cap.read()
        self.frame = frame

    def eval(self):
        self.shouldEval = True
        self.pauseButton.configure(state="disabled")
        self.eval_cap()

    def close_camera(self):
        self.cap.release()
        cv2.destroyAllWindows()


def get_num_classes(server_url, server_num_classes_endpoint):
    """
    Get number of classes that model uses.  Useful for initializing gui elements such as the K slider.
    :return: num classes
    """
    print("Sending request for num_classes: {}".format(id))
    start = time.time()
    response = requests.get(server_url + server_num_classes_endpoint)
    print('request returned {} in {} seconds.'.format(response.status_code, time.time() - start))
    print('response:\n{}'.format(response.json()))

    num_classes = response.json()['num_classes']
    return num_classes


if __name__ == "__main__":
    # we expect, as a hand-shake agreement, that there is a .yml config file in top level of lib/configs directory
    config_dir = os.path.join(os.pardir, 'configs', 'image_classification')
    yaml_path = os.path.join(config_dir, 'inference_livestream_keras.yml')
    with open(yaml_path, "r") as stream:
        config = yaml.load(stream)

    ## collect hyper parameters/args from config
    # NOTE: float() is required to parse any exponentials since YAML sends exponentials as strings
    stream_url = config["stream_url"]
    server_url = config["server_url"]
    server_eval_endpoint = config["server_eval_endpoint"]
    server_num_classes_endpoint = config["server_num_classes_endpoint"]
    dst_dir = config["destination_directory"]

    num_classes = get_num_classes(server_url=server_url, server_num_classes_endpoint=server_num_classes_endpoint)
    cam_manager = webcam_manager(stream_url=stream_url,
                                 server_url=server_url,
                                 server_eval_endpoint=server_eval_endpoint,
                                 server_num_classes_endpoint=server_num_classes_endpoint,
                                 num_classes=num_classes)
    cam_manager.root.mainloop()
    cam_manager.close_camera()