import numpy as np
import cv2 as cv
from glob import glob   

import os, pathlib
# import torch
from utils.timer import Timer
from face_utils import load_faceboxes, get_facebox_coords
                                                        

# ================== OpenCV Video Capture =================== #
cap = cv.VideoCapture(0)
print('Frame width:', int(cap.get(cv.CAP_PROP_FRAME_WIDTH)))
print('Frame height:', int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
print('Capture frame rate:', cap.get(cv.CAP_PROP_FPS))
font = cv.FONT_HERSHEY_SIMPLEX
# =========================================================== #

# Initialize facesboxes model
# net = load_faceboxes()

_t = {'fps': Timer()}

# load 360s
print('loading 360')
jpgs = glob("./data/afv_100/*.jpg")
img_set = []
for j in jpgs:
    img_set.append(cv.imread(j))
print(f"{len(img_set)} frames loaded")

for img in img_set:

    # Image Resize for dev
    outframe = cv.resize(img, None, fx=0.4, fy=0.4)

    cv.imshow('frame', outframe)

    key = cv.waitKey(100)  # pause for 100 ms
    if key == 27:  # if ESC is pressed, exit loop
        cv.destroyAllWindows()
        break