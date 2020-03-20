######## Picamera Object Detection Using Tensorflow Classifier #########

# Description: 
# This program uses a TensorFlow classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a Picamera feed.
# It draws boxes and scores around the objects of interest in each frame from
# the Picamera. It also can be used with a webcam by adding "--usbcam"
# when executing this script from the terminal.

import os
import cv2
import time
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf
import argparse
import sys

# Set up camera constants
IM_WIDTH = 1280
IM_HEIGHT = 720

camera_type = 'picamera'

# This is needed since the working directory is the object_detection folder.
sys.path.append('..')

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 90

## Load the label map.
# Label maps map indices to category names, so that when the convolution
# network predicts `5`, we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize camera and perform object detection.

### Picamera ###
if camera_type == 'picamera':
    # Initialize Picamera and grab reference to the raw capture
    camera = PiCamera()
    camera.resolution = (IM_WIDTH,IM_HEIGHT)
    camera.framerate = 10
    rawCapture = PiRGBArray(camera, size=(IM_WIDTH, IM_HEIGHT))
    rawCapture.truncate(0)

    time.sleep(2)
    camera.capture(rawCapture, format="bgr")
    image = rawCapture.array
        
    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    frame = np.copy(image)
    frame.setflags(write=1)
    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
         [detection_boxes, detection_scores, detection_classes, num_detections],
         feed_dict={image_tensor: frame_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,min_score_thresh = 0.4)

    # Print detected classes on terminal
    ls = [category_index.get(value) for index, value in enumerate(classes[0]) if scores[0, index] > 0.5]
    s = ""
    for i, j in enumerate(ls):
        s += j['name'] + " "
    print(s)

    ###TEXT TO SPEECH###
    if len(s) > 0:
        from subprocess import call

        cmd_beg = 'espeak -v en -k5 -s120 '
        cmd_end = ' | aplay /home/pi/Desktop/Text.wav  2>/dev/null'  # To play back the stored .wav file and to dump the std errors to /dev/null
        cmd_out = '--stdout > /home/pi/Desktop/Text.wav '  # To store the voice file

        # Replacing ' ' with '_' to identify words in the text entered
        s = s.replace(' ', '_')

        # Calls the Espeak TTS Engine to read aloud a Text
        call([cmd_beg + cmd_out + s + cmd_end], shell=True)
        os.system("omxplayer ~/Desktop/Text.wav")

    # All the results have been drawn on the frame, so it's time to display it.
    r = 600.0 / frame.shape[1]
    dim = (600, int(frame.shape[0] * r))

    # perform the actual resizing of the image and show it
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow("Object Detector", resized)

    cv2.waitKey(5000)
    rawCapture.truncate(0)

    camera.close()

cv2.destroyAllWindows()

