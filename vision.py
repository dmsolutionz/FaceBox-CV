import numpy as np
import cv2 as cv
from glob import glob  
from collections import OrderedDict
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
trigger = 0
SPIN = False
last_centroid = 0

# Load 360 into dict
MEDIA_TYPE = "video"

if MEDIA_TYPE == "images":
    jpgs = glob("./data/afv_100/*.jpg")
    vd = OrderedDict()
    for i, j in enumerate(jpgs):
        vd[i] = cv.imread(j)

elif MEDIA_TYPE == "video":
    vid_cap = cv.VideoCapture('data/JC_C_Reshoot_06_single_1080x860.mp4')
    vd = OrderedDict()
    i = 0
    print('\nLoading mp4')
    while(vid_cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = vid_cap.read()
        if ret is True:
            vd[i] = frame
        else:
            break
        i += + 1
print(len(vd), 'frames loaded')

# Initialise starting media frame
index = 0
m_len = len(vd)

while True:
    _t['fps'].tic()
    m_frame = vd[index]

    # Capture frame and mirror horizontal
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)

    # Get bounding boxes from NN foward-pass
    dets = get_facebox_coords(frame, net)

    # Masking Dev only
    cv.rectangle(frame, (0,0), (frame.shape[1],frame.shape[0]), color=(0,0,0), thickness=-1)

    # Add vertical blinds
    y1 = int(frame.shape[1] * 0.2)
    y2 = int(frame.shape[1] * 0.8)
    cv.rectangle(frame, (y1, 0), (y2, frame.shape[0]), color=(25,25,25), thickness=-1) 

    # Loop bounding boxes
    for i, det in enumerate(dets):
        xmin = int(round(det[0]))
        ymin = int(round(det[1]))
        xmax = int(round(det[2]))
        ymax = int(round(det[3]))
        score = det[4]
        # Draw bounding box
        # print(xmin, ymin, xmax, ymax)
        cv.rectangle(frame, (xmin,ymin), (xmax,ymax), color=(188,188,188), thickness=2, lineType=cv.LINE_AA)

        # Event condition on highest-score facebox
        if i == 0:
            centroid = int((xmin + xmax) / 2)
            if (centroid < y2) & (centroid > y1):
                cv.circle(frame, (centroid, ymin-25), 15, color=(30,200,30), thickness=-1, lineType=cv.LINE_AA)

                # -----------------------------------
                # 2. Tracking face movement for 360 control
                pos = centroid - last_centroid

                if abs(pos) > 1:  # Trigger change of media frame if facebox moves > 5 frames
                    index += int(pos/5)
                    index = index % m_len  # take modulus to allow looping
                    m_frame = vd[index]
                # -----------------------------------

            elif trigger == 1:  # reset trigger if bounding box exits area
                trigger = 0
                # pyautogui.moveTo(x, y)

            last_centroid = centroid


    # 
    # Image Resize for dev
    outframe = cv.resize(frame, None, fx=0.4, fy=0.4)
    bw = False
    if bw:
        outframe = cv.cvtColor(outframe, cv.COLOR_BGR2GRAY)

    # Render FPS
    _t['fps'].toc()
    fps = 'FPS: {:.3f}'.format(1 / _t['fps'].diff)
    cv.putText(outframe, fps, (11,23), font, 0.35, (255,255,255), 1, cv.LINE_AA)

    # Display frames
    cv.imshow('Headtracking', outframe)

    m_frame = cv.resize(m_frame, None, fx=0.6, fy=0.6)
    cv.imshow('Media Output', m_frame)


    if cv.waitKey(1) & 0xFF == 27:
        break  # esc to quit
    if cv.waitKey(1) & 0xFF == 'r':
        index = 0  # esc to quit

cap.release()
cv.destroyAllWindows()




