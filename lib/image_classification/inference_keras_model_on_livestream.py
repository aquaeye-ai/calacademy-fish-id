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

SERVER_URL = 'http://172.17.0.2:8000/eval'


class webcam_manager():
    def __init__(self, stream_url=None):
        # init state
        self.shouldPause = False
        self.shouldHideModelControls = False
        self.frame_count = 0
        self.stream_url = stream_url
        self.font = font = cv2.FONT_HERSHEY_SIMPLEX
        self.width, self.height = 800, 600
        self.classification_buffer = []
        self.classification_buffer_index = 0
        self.classification_buffer_ready = False
        self.unsure_threshold = 0.5
        self.drone_threshold = 0.65
        self.payload_threshold = 0.5
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
        # TODO: use Tkinter's grid layout instead of pack for arrangement/structuring
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
        self.init_tk_stopButton()
        self.init_tk_pauseButton()
        self.init_tk_hideShowMCButton()

    def init_tk_scales(self):
        self.init_tk_unsure_thrshld_scale()
        self.init_tk_drone_thrshld_scale()
        self.init_tk_payload_thrshld_scale()

    def init_tk_lFrame(self):
        self.lFrame = tk.LabelFrame(self.root, relief="ridge", borderwidth=4, background="red", text="Frame",
                                    font="bold", labelanchor="n")
        self.lFrame.grid(row=0, column=0, padx=10, pady=10, rowspan=4)
        # self.lFrame.pack(side="left", padx=10, pady=10)

    def init_tk_rFrame(self):
        self.rFrame = tk.Frame(self.root)
        self.rFrame.grid(row=0, column=1, padx=10, pady=10)
        # self.rFrame.pack(side="right", padx=10, pady=10)

    def init_tk_inputsFrame(self):
        self.inputsFrame = tk.LabelFrame(self.root, text="Input", font="bold", labelanchor="n")
        self.inputsFrame.grid(row=0, column=1, padx=10, pady=10)
        # self.inputsFrame.pack(side="top", padx=10, pady=10)

    def init_tk_cmFrame(self):
        self.cmFrame = tk.LabelFrame(self.root, text="Recommmended Counter Measure", font="bold", labelanchor="n")
        self.cmFrame.grid(row=1, column=1, padx=10, pady=10)
        # self.cmFrame.pack(side="top", padx=10, pady=10)

    def init_tk_pcFrame(self):
        self.pcFrame = tk.LabelFrame(self.root, text="Program Controls", font="bold", labelanchor="n")
        # self.pcFrame.pack(side="top", padx=10, pady=10)
        self.pcFrame.grid(row=2, column=1, padx=10, pady=10)

    def init_tk_mcFrame(self):
        self.mcFrame = tk.LabelFrame(self.root, text="Model Controls", font="bold", labelanchor="n")
        # self.mcFrame.pack(side="top", padx=10, pady=10)
        self.mcFrame.grid(row=3, column=1, padx=10, pady=10)

    def init_tk_imageLabel(self):
        self.imageLabel = tk.Label(self.lFrame, relief="sunken")
        # self.imageLabel.pack(padx=2, pady=2)
        self.imageLabel.grid(row=0, column=0, padx=2, pady=2)

    def init_tk_inputsLabel(self):
        self.inputsLabel = tk.Label(self.inputsFrame, anchor="w", width=75, justify="left", relief="ridge", borderwidth=2)
        # self.inputsLabel.pack(padx=10, pady=10)
        self.inputsLabel.grid(row=0, column=0, padx=10, pady=10, columnspan=3)

    def init_tk_cmLabel(self):
        self.cmLabel = tk.Label(self.cmFrame, anchor="w", width=75, justify="left", relief="ridge", borderwidth=2)
        # self.cmLabel.pack(padx=10, pady=10)
        self.cmLabel.grid(row=0, column=0, padx=10, pady=10, columnspan=3)

    def init_tk_jammingButton(self):
        self.jammingButton = tk.Button(self.cmFrame, width=20, relief="raised", borderwidth=2, text="Jam")
        # self.jammingButton.pack(side="left", padx=10, pady=10)
        self.jammingButton.grid(row=1, column=0, padx=10, pady=10)

    def init_tk_empButton(self):
        self.empButton = tk.Button(self.cmFrame, width=20, relief="raised", borderwidth=2, text="EMP")
        # self.empButton.pack(side="left", padx=5, pady=10)
        self.empButton.grid(row=1, column=1, padx=5, pady=10)

    def init_tk_hackButton(self):
        self.hackButton = tk.Button(self.cmFrame, width=20, relief="raised", borderwidth=2, text="Hack")
        # self.hackButton.pack(side="left", padx=10, pady=10)
        self.hackButton.grid(row=1, column=2, padx=10, pady=10)

    def init_tk_startButton(self):
        self.startButton = tk.Button(self.pcFrame, width=13, relief="raised", borderwidth=2, text="Start",
                                     command=self.start)
        # self.startButton.pack(side="left", padx=10, pady=10)
        self.startButton.grid(row=1, column=0, padx=10, pady=10)

    def init_tk_stopButton(self):
        self.stopButton = tk.Button(self.pcFrame, width=13, relief="raised", borderwidth=2, text="Stop",
                                    command=self.stop)
        # self.stopButton.pack(side="left", padx=10, pady=10)
        self.stopButton.grid(row=1, column=1, padx=10, pady=10)

    def init_tk_pauseButton(self):
        self.pauseButton = tk.Button(self.pcFrame, width=13, relief="raised", borderwidth=2, text="Pause",
                                     command=self.pause)
        # self.pauseButton.pack(side="left", padx=10, pady=10)
        self.pauseButton.grid(row=1, column=2, padx=10, pady=10)

    def init_tk_hideShowMCButton(self):
        self.hideMCButton = tk.Button(self.pcFrame, width=13, relief="raised", borderwidth=2,
                                      text="Hide/Show Model Controls", wraplength=90,
                                      command=self.hideShowModelControls)
        # self.hideMCButton.pack(side="left", padx=10, pady=10)
        self.hideMCButton.grid(row=1, column=3, padx=10, pady=10)

    def init_tk_unsure_thrshld_scale(self):
        self.utScale = tk.Scale(self.mcFrame, from_=0, to=1, resolution=0.01, label="UNSURE_THRESHOLD",
                                command=self.utScaleChanged)
        # self.utScale.pack(side="left", padx=10, pady=10)
        self.utScale.grid(row=0, column=0, padx=10, pady=10)
        self.utScale.set(self.unsure_threshold)

    def init_tk_drone_thrshld_scale(self):
        self.dtScale = tk.Scale(self.mcFrame, from_=0, to=1, resolution=0.01, label="DRONE_THRESHOLD",
                                command=self.dtScaleChanged)
        # self.dtScale.pack(side="left", padx=10, pady=10)
        self.dtScale.grid(row=0, column=1, padx=10, pady=10)
        self.dtScale.set(self.drone_threshold)

    def init_tk_payload_thrshld_scale(self):
        self.ptScale = tk.Scale(self.mcFrame, from_=0, to=1, resolution=0.01, label="PAYLOAD_THRESHOLD",
                                command=self.ptScaleChanged)
        # self.ptScale.pack(side="left", padx=10, pady=10)
        self.ptScale.grid(row=0, column=2, padx=10, pady=10)
        self.ptScale.set(self.payload_threshold)

    def utScaleChanged(self, event):
        self.unsure_threshold = self.utScale.get()

    def dtScaleChanged(self, event):
        self.drone_threshold = self.dtScale.get()

    def ptScaleChanged(self, event):
        self.payload_threshold = self.ptScale.get()

    def hideShowModelControls(self):
        self.shouldHideModelControls = ~self.shouldHideModelControls
        if self.shouldHideModelControls:
            # self.mcFrame.pack_forget()
            self.mcFrame.grid_remove()
        else:
            # self.mcFrame.pack()
            self.mcFrame.grid()

    def display_cap(self):
        """Summary: Display frames from the camera until q is pressed.
        """
        while (True):
            # Capture frame-by-frame
            ret, frame = self.cap.read()

            # Display the resulting frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def write_cap(self, frame_count=300):
        """Summary: Record a stream of frasmes (no evaluation)
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

    def eval_cap(self):
        """Summary: Run and evaluate a stream of frames from the camera.
        """
        if not self.shouldPause:
            i = self.frame_count
            ret, frame = self.cap.read()

            frame_counter = str(i)

            img_str = cv2.imencode('.jpg', frame)[1].tostring()

            # response = self.eval_image(i, img_str)

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
            # if response['drone_class']:
            #     inputsLabelText += "Radar: Geolocation -> {lat: 37.4712310, lon: -122.1324510, elevation: 10m}\n"
            #     inputsLabelText += "Visual: Type of UAV -> {} Quadcopter\n".format(response['drone_class'])
            #     inputsLabelText += self.drone_specs(response['drone_class'])
            # else:
            #     inputsLabelText += "\nRadar: Geolocation\nVisual: No drone detected"

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
            self.cmLabel.after(1, self.eval_cap)
            self.frame_count += 1

    def drone_specs(self, drone=""):
        """Summary: return drone specs for drone in text block with each line corresponding to key spec
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

    def eval_image(self, image_id, image_data):
        """Summary: Evaluate a single frame. Sends data to model server api and retrieves resposne.

           TODO: Currently runs about 0.05 seconds round trip. Look for where this could be sped up.
        """
        json_payload = json.dumps({'image_id': image_id,
                                   'image_data': base64.b64encode(image_data)})

        print("Sending request - image_id: {}".format(image_id))
        start = time.time()
        response = requests.post(SERVER_URL, json=json_payload)
        print('request returned {} in {} seconds.'.format(response.status_code, time.time() - start))
        # print(response.json())

        parsed_response = self.parse_response(response.json(), image_id)
        print(parsed_response)
        return parsed_response

    def parse_response(self, response, image_id):
        """Summary: Takes the json returned from the model api for a evaluated frame and returns
                classification broken down by drone, payload with confidence, thresholding and
                smoothing over X frames.

        """

        # Put threshold on invdividual classifications, if not over the mark, put Unsure, best guess: {} with {}% confidence
        UNSURE_THRESHOLD = self.unsure_threshold
        DRONE_THRESHOLD = self.drone_threshold
        PAYLOAD_THRESHOLD = self.payload_threshold

        CLASSIFICATION_BUFFER_LENGTH = 10

        # Parse the evauluation string to get classes and scores.
        # TODO: Return json from model we can use directly as a dictionary rather than parse.
        classifications = response['evaluation'].split('|')
        classifications = [classification.split('(') for classification in classifications]

        _classes = [classification[0].strip() for classification in classifications[:-1]]
        scores = [classification[1].strip() for classification in classifications[:-1]]
        scores = [float(score[8:-1]) for score in scores]

        # Turn classes/scores list into dict of class: score
        class_scores = {_class: scores[index] for index, _class in enumerate(_classes)}

        # For first ten, don't smooth, just fill buffer, otherwise update the buffer
        if self.classification_buffer_ready:
            self.classification_buffer[self.classification_buffer_index] = class_scores
            self.classification_buffer_index += 1
        else:
            self.classification_buffer.append(class_scores)
            self.classification_buffer_index += 1
            if self.classification_buffer_index >= CLASSIFICATION_BUFFER_LENGTH:
                self.classification_buffer_ready = True

        if self.classification_buffer_index >= CLASSIFICATION_BUFFER_LENGTH:
            self.classification_buffer_index = 0
        # Compute the average

        if self.classification_buffer_ready:
            counter_sum = sum((Counter(dict(x)) for x in self.classification_buffer), Counter())
            class_scores = {key: value / (1.0 * CLASSIFICATION_BUFFER_LENGTH) for key, value in counter_sum.iteritems()}

        _classes = []
        scores = []
        for key, value in class_scores.iteritems():
            _classes.append(key)
            scores.append(value)

        phantom_text = ['phantom']
        phantom_score = 0
        parrot_text = ['parrot']
        parrot_score = 0
        no_drone_text = ['no drone']
        no_drone_score = 0

        payload_text = ['camera', 'package']
        payload_score = 0
        no_payload_text = ['no payload']
        no_payload_score = 0

        camera_text = ['camera']
        camera_score = 0
        unknown_package_text = ['package']
        unknown_package_score = 0

        for index, _class in enumerate(_classes):
            if any(text in _class for text in phantom_text):
                phantom_score += scores[index]
            if any(text in _class for text in parrot_text):
                parrot_score += scores[index]
            if any(text in _class for text in no_drone_text):
                no_drone_score += scores[index]

            if any(text in _class for text in payload_text):
                payload_score += scores[index]
            if any(text in _class for text in no_payload_text):
                no_payload_score += scores[index]

            if any(text in _class for text in camera_text):
                camera_score += scores[index]
            if any(text in _class for text in unknown_package_text):
                unknown_package_score += scores[index]

        # Pick drone/no drone
        drone_class = False
        drone_score = no_drone_score
        if (phantom_score > DRONE_THRESHOLD):
            drone_class = 'phantom'
            drone_score = phantom_score
        if (parrot_score > DRONE_THRESHOLD):
            drone_class = 'parrot'
            drone_score = parrot_score

        if drone_class:
            if drone_score < UNSURE_THRESHOLD:
                drone_class = 'unsure, best guess is {}'.format(drone_class)

        # Pick payload/no payload. if no_payload, set score.
        payload = False
        if (payload_score > PAYLOAD_THRESHOLD):
            payload = True
        if no_payload_score > payload_score:
            payload = False
            payload_score = no_payload_score

        # If payload, pick payload type and set score.
        payload_type = False
        if payload:
            if camera_score > unknown_package_score:
                payload_type = 'camera'
                payload_score = camera_score
            else:
                payload_type = 'unknown package'
                payload_score = unknown_package_score

        if payload:
            if payload_score < UNSURE_THRESHOLD:
                payload_type = 'unsure, best guess is {}'.format(payload_type)

        # Create response dict:
        parsed_response = {'drone_class': drone_class,
                           'drone_score': drone_score,
                           'payload': payload,
                           'payload_score': payload_score,
                           'payload_type': payload_type}

        return parsed_response

    def run(self):
        self.eval_cap()
        # self.display_cap()

    def start(self):
        self.shouldPause = False
        self.run()
        self.startButton.configure(state="disabled")

    def stop(self):
        self.root.quit()

    def pause(self):
        self.shouldPause = True
        self.startButton.configure(state="normal")

    def close_camera(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # we expect, as a hand-shake agreement, that there is a .yml config file in top level of lib/configs directory
    config_dir = os.path.join(os.pardir, 'configs', 'image_classification')
    yaml_path = os.path.join(config_dir, 'inference_keras_model_on_livestream.yml')
    with open(yaml_path, "r") as stream:
        config = yaml.load(stream)

    ## collect hyper parameters/args from config
    # NOTE: float() is required to parse any exponentials since YAML sends exponentials as strings
    stream_url = config["stream_url"]
    dst_dir = config["destination_directory"]

    cam_manager = webcam_manager(stream_url=stream_url)
    cam_manager.root.mainloop()
    cam_manager.close_camera()