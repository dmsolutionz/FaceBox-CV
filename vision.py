import cv2
from glob import glob  
from collections import OrderedDict
import os, pathlib
import torch

from utils.timer import Timer
from face_utils import load_faceboxes, get_facebox_coords
from grid_display import grdifiy_four


# ================== OpenCV Video Capture =================== #
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('Frame width:', frame_width)
print('Frame height:', frame_height)
print('Capture frame rate:', cap.get(cv2.CAP_PROP_FPS))
font = cv2.FONT_HERSHEY_SIMPLEX
# =========================================================== #

# Initialize facesboxes model
net = load_faceboxes()
_t = {'fps': Timer()}
_nn = {'fps': Timer()}
trigger = 0
SPIN = False
last_centroid = 0

# Load 360 into dict
MEDIA_TYPE = "video"

if MEDIA_TYPE == "images":
    jpgs = glob("./data/afv_100/*.jpg")
    vd = OrderedDict()
    for i, j in enumerate(jpgs):
        vd[i] = cv2.imread(j)

elif MEDIA_TYPE == "video":
    vid_path = 'data/JC_06_single_1080x860_4x.mp4'
    vid_cap = cv2.VideoCapture(vid_path)
    vd = OrderedDict()
    i = 0
    print('\nLoading media:', vid_path)
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
home_index = 108
m_len = len(vd)
col_index = 0

while True:
    _t['fps'].tic()
    m_frame = vd[index]
    inside_activated = False

    # Capture frame and mirror horizontal
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # CROP TO CENTER IMAGE for passing to nn
    # c1 = frame_height/4
    # c2 = frame_width/4
    # y1, y2 = int(c1), int(frame_height/2 + c1)
    # x1, x2 = int(c2), int(frame_width/2 + c2)
    # frame_nn = cv2.resize(frame[y1:y2, x1:x2], None, fx=0.5, fy=0.5)

    frame = cv2.resize(frame, None, fx=0.3, fy=0.3)
    dets = get_facebox_coords(frame, net)
    
    # Masking Dev only
    cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), color=(25,25,25), thickness=-1)

    # Add vertical blinds
    y1 = int(frame.shape[1] * 0.0)
    y2 = int(frame.shape[1] * 1.0)
    cv2.rectangle(frame, (y1, 0), (y2, frame.shape[0]), color=(0,0,0), thickness=-1)
    # cv2.rectangle(frame, (y2, 0), (y2, frame.shape[0]), color=(0,0,0), thickness=-1)
    
    # Loop bounding boxes
    for i, det in enumerate(dets):
        xmin = int(round(det[0]))
        ymin = int(round(det[1]))
        xmax = int(round(det[2]))
        ymax = int(round(det[3]))
        score = det[4]
        # Draw bounding box
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color=(188,188,188), thickness=2, lineType=cv2.LINE_AA)

        # Event condition on highest-score facebox
        if i == 0:
            centroid = int((xmin + xmax) / 2)
            if (centroid < y2) & (centroid > y1):
                inside_activated = True
                cv2.circle(frame, (centroid, ymin-25), 10, color=(30,200,30), thickness=-1, lineType=cv2.LINE_AA)

                # -----------------------------------
                # 2. Tracking face movement for 360 control
                pos = centroid - last_centroid

                if abs(pos) > 0:  # Trigger change of media frame if facebox position moves
                    index += int(pos*2)
                    index = index % (m_len - 1)  # take modulus to allow looping
                    m_frame = vd[index]
                # -----------------------------------

            last_centroid = centroid

    if inside_activated == False:  # Spin slowly if no face is in frame
        index += 4
        index = index % (m_len - 1)
        m_frame = vd[index]
    
    # Image Resize for dev
    outframe = cv2.resize(frame, None, fx=1.5, fy=1.5)
    # outframe = frame
    bw = False
    if bw:
        outframe = cv2.cvtColor(outframe, cv2.COLOR_BGR2GRAY)

    # Render FPS
    _t['fps'].toc()
    fps = 'FPS: {:.3f}'.format(1 / _t['fps'].diff)
    cv2.putText(outframe, fps, (11,23), font, 0.35, (255,255,255), 1, cv2.LINE_AA)

    # Display headtracking frame
    cv2.imshow('Headtracking', outframe)
    
    # Change to HSV and mess with hue value
    # m_frame = cv2.cvtColor(m_frame, cv2.COLOR_BGR2HSV)
    # m_frame[:,:,2]
    # m_frame[:,:,:] = col_index % 255

    m_frame = cv2.resize(m_frame, None, fx=0.5, fy=0.5)
    four_up = grdifiy_four(m_frame)
    cv2.imshow('Media Output', four_up)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # ESC TO QUIT
        break
    elif k == ord('r'):   # RESET ORIENTATION
        print('Key press: R\t Resetting Orientation')
        index = home_index
    elif k == ord('t'):
        print('Key press: T\t Home index increased +12 frames')
        home_index += 12


cap.release()
cv2.destroyAllWindows()
