# USAGE
# python pi_face_recognition.py --cascade haarcascade_frontalface_default.xml --encodings encodings.pickle

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import argparse
import imutils
import os
import pickle
import time
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera

# Set up camera constants
IM_WIDTH = 1280
IM_HEIGHT = 720
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
	help = "path to where the face cascade resides")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
args = vars(ap.parse_args())

# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(args["encodings"], "rb").read())
detector = cv2.CascadeClassifier(args["cascade"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
#vs = VideoStream(usePiCamera=True).start()
#time.sleep(2.0)

# start the FPS counter
#fps = FPS().start()

# loop over frames from the video file stream

# grab the frame from the threaded video stream and resize it
# to 500px (to speedup processing)
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
frame = imutils.resize(frame, width=500)

# convert the input frame from (1) BGR to grayscale (for face
# detection) and (2) from BGR to RGB (for face recognition)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# detect faces in the grayscale frame
rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
	minNeighbors=5, minSize=(30, 30),
	flags=cv2.CASCADE_SCALE_IMAGE)

# OpenCV returns bounding box coordinates in (x, y, w, h) order
# but we need them in (top, right, bottom, left) order, so we
# need to do a bit of reordering
boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

# compute the facial embeddings for each face bounding box
encodings = face_recognition.face_encodings(rgb, boxes)
names = []

# loop over the facial embeddings
for encoding in encodings:
	# attempt to match each face in the input image to our known
	# encodings
	matches = face_recognition.compare_faces(data["encodings"],
		encoding)
	name = "Unknown"

	# check to see if we have found a match
	if True in matches:
		# find the indexes of all matched faces then initialize a
		# dictionary to count the total number of times each face
		# was matched
		matchedIdxs = [i for (i, b) in enumerate(matches) if b]
		counts = {}

		# loop over the matched indexes and maintain a count for
		# each recognized face face
		for i in matchedIdxs:
			name = data["names"][i]
			counts[name] = counts.get(name, 0) + 1

		# determine the recognized face with the largest number
		# of votes (note: in the event of an unlikely tie Python
		# will select first entry in the dictionary)
		name = max(counts, key=counts.get)
		print(name)
		if len(name) > 0:
			from subprocess import call

			cmd_beg = 'espeak -v en -k5 -s120 '
			cmd_end = ' | aplay /home/pi/Desktop/audio.wav  2>/dev/null'  # To play back the stored .wav file and to dump the std errors to /dev/null
			cmd_out = '--stdout > /home/pi/Desktop/audio.wav '  # To store the voice file

			# Calls the Espeak TTS Engine to read aloud a Text
			call([cmd_beg + cmd_out + name + cmd_end], shell=True)
			os.system("omxplayer ~/Desktop/audio.wav")
	# update the list of names
	names.append(name)

# loop over the recognized faces
for ((top, right, bottom, left), name) in zip(boxes, names):
	# draw the predicted face name on the image
	cv2.rectangle(frame, (left, top), (right, bottom),
		(0, 255, 0), 2)
	y = top - 15 if top - 15 > 15 else top + 15
	cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
		0.75, (0, 255, 0), 2)

# display the image to our screen
cv2.imshow("Frame", frame)
key = cv2.waitKey(5000) & 0xFF

cv2.destroyAllWindows()
