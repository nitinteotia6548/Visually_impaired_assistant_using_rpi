from pad4pi import rpi_gpio
import time
import os
from subprocess import call


cmd_beg = 'espeak -v en -k5 -s120 '
cmd_end = ' | aplay /home/pi/Desktop/audio.wav  2>/dev/null'  # To play back the stored .wav file and to dump the std errors to /dev/null
cmd_out = '--stdout > /home/pi/Desktop/audio.wav '  # To store the voice file

# Replacing ' ' with '_' to identify words in the text entered
a = 'Hello I am here to assit you'
a = a.replace(' ', '_')

# Calls the Espeak TTS Engine to read aloud a Text
call([cmd_beg + cmd_out + a + cmd_end], shell=True)
os.system("omxplayer ~/Desktop/audio.wav")

# Setup Keypad
KEYPAD = [
        ["1","2","3","A"],
        ["4","5","6","B"],
        ["7","8","9","C"],
        ["*","0","#","D"]
]

# same as calling: factory.create_4_by_4_keypad, still we put here fyi:
ROW_PINS = [21, 20, 16, 26] # BCM numbering
COL_PINS = [19, 13, 6, 5] # BCM numbering

factory = rpi_gpio.KeypadFactory()

# Try factory.create_4_by_3_keypad
# and factory.create_4_by_4_keypad for reasonable defaults
keypad = factory.create_keypad(keypad=KEYPAD, row_pins=ROW_PINS, col_pins=COL_PINS)

#keypad.cleanup()

def printKey(key):
    print(key)
    if key=='A':
        cmd_beg = 'espeak -v en -k5 -s120 '
        cmd_end = ' | aplay /home/pi/Desktop/audio.wav  2>/dev/null'  # To play back the stored .wav file and to dump the std errors to /dev/null
        cmd_out = '--stdout > /home/pi/Desktop/audio.wav '  # To store the voice file

        # Replacing ' ' with '_' to identify words in the text entered
        a = 'Starting Object Detection'
        a = a.replace(' ', '_')

        # Calls the Espeak TTS Engine to read aloud a Text
        call([cmd_beg + cmd_out + a + cmd_end], shell=True)
        os.system("omxplayer ~/Desktop/audio.wav")
        os.chdir(r"/home/pi/tensorflow1/models/research/object_detection")
        os.system("python3 Object_detection_picamera2.py")

    if key=='B':
        cmd_beg = 'espeak -v en -k5 -s120 '
        cmd_end = ' | aplay /home/pi/Desktop/audio.wav  2>/dev/null'  # To play back the stored .wav file and to dump the std errors to /dev/null
        cmd_out = '--stdout > /home/pi/Desktop/audio.wav '  # To store the voice file

        # Replacing ' ' with '_' to identify words in the text entered
        a = 'Starting Face Recognition'
        a = a.replace(' ', '_')

        # Calls the Espeak TTS Engine to read aloud a Text
        call([cmd_beg + cmd_out + a + cmd_end], shell=True)
        os.system("omxplayer ~/Desktop/audio.wav")
        os.chdir(r"/home/pi/Desktop/Raspberry_Pi/pi-face-recognition")
        os.system("python3 pi_face_recognition.py --cascade haarcascade_frontalface_default.xml \--encodings encodings.pickle")



# printKey will be called each time a keypad button is pressed
keypad.registerKeyPressHandler(printKey)

try:
  while(True):
    time.sleep(0.2)
except:
 keypad.cleanup()