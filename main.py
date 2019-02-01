import numpy as np
import cv2 as cv

import os, pathlib
import torch
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
net = load_faceboxes()

_t = {'fps': Timer()}

while True:
    _t['fps'].tic()

    # Capture frame and mirror horizontal
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)

    # Get bounding boxes from NN foward-pass
    dets = get_facebox_coords(frame, net)

    # Masking Dev only
    cv.rectangle(frame, (0,0), (frame.shape[1],frame.shape[0]), color=(0,0,0), thickness=-1)

    # Add vertical blinds
    y1 = int(frame.shape[1] * 0.4)
    y2 = int(frame.shape[1] * 0.6)
    cv.rectangle(frame, (y1, 0), (y2, frame.shape[0]), color=(25,25,25), thickness=-1)

    # Loop bounding boxes
    for det in dets:
        xmin = int(round(det[0]))
        ymin = int(round(det[1]))
        xmax = int(round(det[2]))
        ymax = int(round(det[3]))
        score = det[4]
        print(xmin, ymin, xmax, ymax)
        # Draw bounding box
        cv.rectangle(frame, (xmin,ymin), (xmax,ymax), color=(188,188,188), thickness=2, lineType=cv.LINE_AA)

        # Event condition
        centroid = int((xmin + xmax) / 2)
        if (centroid < y2) & (centroid > y1):
            cv.circle(frame, (centroid, ymin-25), 15, color=(30,200,30), thickness=-1, lineType=cv.LINE_AA)

    # Image operations dev
    outframe = cv.resize(frame, None, fx=0.4, fy=0.4)
    # outframe = cv.cvtColor(outframe, cv.COLOR_BGR2GRAY)

    # Render FPS
    _t['fps'].toc()
    fps = 'FPS: {:.3f}'.format(1 / _t['fps'].diff)
    cv.putText(outframe, fps, (11,23), font, 0.35, (255,255,255), 1, cv.LINE_AA)

    # Display frame
    cv.imshow('frame', outframe)

    if cv.waitKey(1) & 0xFF == 27:
        break  # esc to quit

cap.release()
cv.destroyAllWindows()




