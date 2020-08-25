import os
import cv2
import json
import yaml
import time
import base64
import requests
import datetime
import Image, ImageTk

import Tkinter as tk
import lib.file_utils as fu

from PIL import Image, ImageTk
from collections import Counter


class webcam_manager():
    def __init__(self, stream_url=None, server_url=None, server_eval_endpoint=None, server_num_classes_endpoint=None,
                 num_classes=None, thumbnails_dir=None):
        # init state
        self.thumbnails_dir = thumbnails_dir
        self.rect = None
        self.hasRect = False
        self.shouldPause = False
        self.shouldHideModelControls = False
        self.frame_count = 0
        self.frame = None
        self.stream_url = stream_url
        self.server_url = server_url
        self.server_eval_endpoint = server_eval_endpoint
        self.server_num_classes_endpoint = server_num_classes_endpoint
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.thumbnail_width, self.thumbnail_height = 100, 100
        self.frame_width, self.frame_height = 1920, 1080 # size of image gathered by camera
        self.img_reduction_factor = 2 # factor by which to reduce image gathered by camera
        self.display_width, self.display_height =  float(self.frame_width) / self.img_reduction_factor, float(self.frame_height) / self.img_reduction_factor # reduce 1920x1080 by factor of 2
        self.classification_buffer = []
        self.classification_buffer_index = 0
        self.classification_buffer_ready = False
        self.unsure_threshold = 0.5
        self.K = 1
        self.num_classes = num_classes
        self.common_group_names = {
            'stingrays': {
                'species': ['Rhinoptera javanica', 'Taeniura lymma', 'Himantura uarnak', 'Neotrygon kuhlii'],
                'status': "Data deficient - Threatened",
                'diet': "Mollusks, worms, shrimp, clams, crabs, bivalves, gastropods, jellyfish, bony fishes",
                'reproduction': "Viviparous - Ovoviviparous"
            },
            'moonyfishes': {
                'species': ['Monodactylus argenteus'],
                'status': "not yet assessed",
                'diet': "Plankton and detritus",
                'reproduction': "Broadcast spawners; males and females shed gametes into the water, where fertilization occurs"
            },
            'surgeonfishes': {
                'species': ['Acanthurus triostegus'],
                'status': "Least Concern",
                'diet': "Benthic algae",
                'reproduction': "Oviparous broadcast spawners; found in large groups (up to several hundred) that exhibit mass spawning behavior"
            },
            'butterflyfishes': {
                'species': ['Chelmon rostratus'],
                'status': "Least concern",
                'diet': "Benthic invertebrates, which it finds in rock cervices with its elongated snout",
                'reproduction': "Oviparous"
            },
            'pompanos': {
                'species': ['Trachinotus mookalee'],
                'status': "Not yet assessed",
                'diet': "Small fishes and crustaceans",
                'reproduction': "Broadcast spawners"
            },
            'other': {
                'species': ['NA'],
                'status': "NA",
                'diet': "NA",
                'reproduction': "NA"
            }
        }

        # init UI elements
        self.init_cap()
        self.init_tk_root()
        self.init_tk_frames()
        self.init_tk_labels()
        self.init_tk_buttons()
        self.init_tk_scales()
        self.init_draw_canvas()

    def init_cap(self):
        self.cap = cv2.VideoCapture(self.stream_url)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.display_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.display_height)

    def init_tk_root(self):
        self.root = tk.Tk()
        self.root.title("Reef Lagoon")
        self.root.configure(background='gray')
        self.root.bind('<Escape>', lambda e: self.root.quit())

    def init_draw_canvas(self):
        self.canvas = tk.Canvas(self.lFrame, cursor="cross")
        self.canvas.grid(row=0, column=0)
        self.canvas.config(scrollregion=(0, 0, self.display_width, self.display_height), width=self.display_width, height=self.display_height)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_left_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move_left_button_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_left_button_release)
        self.canvas.grid_remove()

    def init_tk_frames(self):
        self.init_tk_lFrame()
        self.init_tk_rFrame()
        self.init_tk_predictionsFrame()
        self.init_tk_exhibitFrame()
        self.init_tk_pcFrame()
        self.init_tk_mcFrame()

    def init_tk_labels(self):
        self.init_tk_imageLabel()
        self.init_tk_predictionsLabel()
        self.init_tk_exhibitLabels(db_name="common_group_names")

    def init_tk_buttons(self):
        self.init_tk_startButton()
        self.init_tk_quitButton()
        self.init_tk_pauseButton()
        self.init_tk_evalButton()
        self.init_tk_undoButton()
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

    def init_tk_predictionsFrame(self):
        self.predictionsFrame = tk.LabelFrame(self.root, text="Predictions", font="bold", labelanchor="n")
        self.predictionsFrame.grid(row=0, column=1, padx=10, pady=10)

    def init_tk_exhibitFrame(self):
        self.exhibitFrame = tk.LabelFrame(self.root, text="In This Exhibit", font="bold", labelanchor="n")
        self.exhibitFrame.grid(row=1, column=1, padx=10, pady=10)

    def init_tk_pcFrame(self):
        self.pcFrame = tk.LabelFrame(self.root, text="Program Controls", font="bold", labelanchor="n")
        self.pcFrame.grid(row=2, column=1, padx=10, pady=10)

    def init_tk_mcFrame(self):
        self.mcFrame = tk.LabelFrame(self.root, text="Model Controls", font="bold", labelanchor="n")
        self.mcFrame.grid(row=3, column=1, padx=10, pady=10)

    def init_tk_imageLabel(self):
        self.imageLabel = tk.Label(self.lFrame, relief="sunken")
        self.imageLabel.grid(row=0, column=0, padx=2, pady=2)

    def init_tk_predictionsLabel(self):
        self.predictionsLabel = tk.Label(self.predictionsFrame, anchor="w", width=75, justify="left", relief="ridge", borderwidth=2)
        self.predictionsLabel.grid(row=0, column=0, padx=10, pady=10, columnspan=3)

    def init_tk_exhibitLabels(self, db_name=None):
        thumbnails = fu.find_files(self.thumbnails_dir, extension='.jpg')
        [thumbnails.append(img) for img in fu.find_files(self.thumbnails_dir, extension='.png')]
        db = getattr(self, db_name)
        for idx, name in enumerate(db.keys()):
            # create frame to hold label so that we can have more visible text
            setattr(self, "{}Frame".format(name), tk.LabelFrame(self.exhibitFrame, relief="ridge", borderwidth=4,
                                                                background="gray", text="{}".format(name), font="bold",
                                                                labelanchor="n"))
            getattr(self, "{}Frame".format(name)).grid(row=0, column=idx, padx=10, pady=10)

            # create label to hold image
            setattr(self, "{}Label".format(name), tk.Label(getattr(self, "{}Frame".format(name)),
                                                           anchor="center",
                                                           width=self.thumbnail_width,
                                                           justify="left",
                                                           relief="ridge",
                                                           borderwidth=10,
                                                           compound=tk.CENTER,
                                                           bg="gray") # color of the perimeter
                    )
            getattr(self, "{}Label".format(name)).grid(row=0, column=0, padx=2, pady=2)

            # set the appropriate thumbnail for the label
            for tn in thumbnails:
                tn_basename = os.path.basename(tn)
                if name == tn_basename[:-4]:
                    setattr(self, "{}Label_tn".format(name), ImageTk.PhotoImage(Image.open(tn).resize((self.thumbnail_height, self.thumbnail_width))))
                    getattr(self, "{}Label".format(name)).configure(image=getattr(self, "{}Label_tn".format(name)))

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
        self.pauseButton.configure(state="disabled")

    def init_tk_evalButton(self):
        self.evalButton = tk.Button(self.pcFrame, width=13, relief="raised", borderwidth=2, text="Predict",
                                    command=self.eval)
        self.evalButton.grid(row=2, column=0, padx=10, pady=10)
        self.evalButton.configure(state="disabled")

    def init_tk_undoButton(self):
        self.undoButton = tk.Button(self.pcFrame, width=13, relief="raised", borderwidth=2, text="Undo",
                                    command=self.undo)
        self.undoButton.grid(row=2, column=1, padx=10, pady=10)
        self.undoButton.configure(state="disabled")

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
        self.kScale = tk.Scale(self.mcFrame, from_=1, to=self.num_classes, resolution=1, label="K: How Many Predictions to Return?", command=self.kScaleChanged)
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
            cv2image = cv2.resize(cv2image, dsize=(int(self.display_width), int(self.display_height)))

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

    def eval_cap(self):
        """
        Run and evaluate a stream of frames from the camera.
        :return: None
        """
        if self.shouldEval:
            # frame = self.frame
            frame = cv2.flip(self.frame, 1)
            print("image shape: {}".format(frame.shape))
            print("image dtype: {}".format(frame.dtype))

            # scale coordinates of selection to match original 1920x1080 frame instead of downsized frame
            rect_coords = self.canvas.coords(self.rect)
            rect_coords_adjusted = tuple([coord * self.img_reduction_factor for coord in rect_coords])

            # cut selection from frame for inferencing on
            x1 = int(rect_coords_adjusted[0])
            y1 = int(rect_coords_adjusted[1])
            x2 = int(rect_coords_adjusted[2])
            y2 = int(rect_coords_adjusted[3])
            selection_frame = frame[y1:y2, x1:x2]

            # cv2.imshow('Selection', selection_frame)
            # cv2.imshow('Frame', frame)
            # cv2.waitKey(0)

            retval, buffer = cv2.imencode('.jpg', selection_frame)

            response = self.eval_image(id=self.frame_count, shape=selection_frame.shape, data=buffer)

            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

            predictionsLabelText = "Timestamp: {}\n".format(st)

            # write top K classes and scores
            predictionsLabelText += "\nTop {} Classes:\n".format(self.K)
            predictionsLabelText += "-----------------------\n"
            for idx, _class in enumerate(response['top_k_classes']):
                predictionsLabelText += "{}: {} -> {:.2f}%\n".format(_class, response['top_k_scores'][idx], response['top_k_scores'][idx] * 100)

            # write all classes and their average scores
            predictionsLabelText += "\nAverage Scores for All Classes:\n"
            predictionsLabelText += "-----------------------\n"
            for idx, average_class in enumerate(response['average_classes']):
                predictionsLabelText += "{}: {}\n".format(average_class, response['average_scores'][idx])

            top_class = response["top_k_classes"][0]

            # display common grouping info for top prediction
            predictionsLabelText += "\nFun Facts for species under {}\n".format(top_class)
            predictionsLabelText += "-----------------------\n"
            predictionsLabelText += "Species: {}\n".format(", ".join(self.common_group_names[top_class]['species']))
            predictionsLabelText += "Status: {}\n".format(self.common_group_names[top_class]['status'])
            predictionsLabelText += "Diet: {}\n".format(self.common_group_names[top_class]['diet'])
            predictionsLabelText += "Reproduction: {}".format(self.common_group_names[top_class]['reproduction'])

            # highlight the top prediction's thumbnail
            getattr(self, "{}Label".format(top_class)).config(bg="green")

            self.predictionsLabel.text = predictionsLabelText
            self.predictionsLabel.configure(text=predictionsLabelText)

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
        # unhighlight any previous prediction results
        for _class in self.common_group_names.keys():
            getattr(self, "{}Label".format(_class)).config(bg="gray")

        # show the video feed and hide the canvas
        self.imageLabel.grid()
        self.canvas.grid_remove()

        # remove any drawn rects
        self.canvas.delete(self.rect)
        self.rect = None
        self.hasRect = False

        self.shouldPause = False
        self.startButton.configure(state="disabled")
        self.evalButton.configure(state="disabled")
        self.undoButton.configure(state="disabled")
        self.pauseButton.configure(state="normal")

        self.run()

    def quit(self):
        self.root.quit()

    def pause(self):
        self.shouldPause = True
        self.pauseButton.configure(state="disabled")
        self.startButton.configure(state="normal")

        # hide the video feed and show the canvas
        self.imageLabel.grid_remove()
        self.canvas.grid()

        # collect the frame for the canvas
        ret, frame = self.cap.read()
        self.frame = frame
        frame = cv2.flip(self.frame, 1)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        cv2image = cv2.resize(cv2image, dsize=(int(self.display_width), int(self.display_height)))
        img = Image.fromarray(cv2image)

        # prevents the newly created image object from being garbage collected...this would otherise lead the canvas to be empty
        self.img_tk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor="nw", image=self.img_tk)

        # initialize drawing
        self.x = self.y = 0
        self.start_x = self.start_y = None

    def eval(self):
        self.shouldEval = True
        self.pauseButton.configure(state="disabled")
        self.eval_cap()

    def undo(self):
        # unhighlight any previous prediction results
        for _class in self.common_group_names.keys():
            getattr(self, "{}Label".format(_class)).config(bg="gray")

        self.canvas.delete(self.rect)
        self.rect = None
        self.hasRect = False

    def close_camera(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def on_mouse_left_button_press(self, event):
        # only allow to draw/inference one rect at a time
        if self.hasRect == False:
            # save mouse drag start position
            self.start_x = event.x
            self.start_y = event.y

            # create rectangle if not yet exist
            self.rect = self.canvas.create_rectangle(self.x, self.y, 1, 1, fill="", outline="#F00", dash=(4, 2))

    def on_mouse_left_button_release(self, event):
        self.evalButton.configure(state="normal")
        self.undoButton.configure(state="normal")
        self.hasRect = True

    def on_mouse_move_left_button_press(self, event):
        # only allow to draw/inference one rect at a time
        if self.hasRect == False:
            curX, curY = (event.x, event.y)

            # expand rectangle as you drag the mouse
            self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)


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
    thumbnails_dir = config["thumbnails_directory"]
    dst_dir = config["destination_directory"]

    num_classes = get_num_classes(server_url=server_url,
                                  server_num_classes_endpoint=server_num_classes_endpoint)
    cam_manager = webcam_manager(stream_url=stream_url,
                                 server_url=server_url,
                                 server_eval_endpoint=server_eval_endpoint,
                                 server_num_classes_endpoint=server_num_classes_endpoint,
                                 num_classes=num_classes,
                                 thumbnails_dir=thumbnails_dir)
    cam_manager.root.mainloop()
    cam_manager.close_camera()