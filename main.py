import cv2 as cv
import numpy as np
import os
from time import time
from windowcapture import WindowCapture
from vision import Vision


# initialize the WindowCapture class
wincap = WindowCapture('METIN2')

# load the trained model
cascade_fish = cv.CascadeClassifier('cascade.xml')
# load an empty Vision class
vision_fish = Vision()

loop_time = time()
while(True):

    # get an updated image of the game
    screenshot = wincap.get_screenshot()

    # do object detection
    rectangles = cascade_fish.detectMultiScale(screenshot)

    # draw the detection results onto the original image
    detection_image = vision_fish.draw_rectangles(screenshot, rectangles)

    # display the images
    cv.imshow('Matches', detection_image)

    # debug the loop rate
    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit.
    key = cv.waitKey(1)
    if key == ord('q'):
        cv.destroyAllWindows()
        break

print('Done.')
