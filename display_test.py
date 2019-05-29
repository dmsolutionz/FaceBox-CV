import cv2
from collections import OrderedDict

# Create VideoCapture object from input file
cap = cv2.VideoCapture('data/JC_C_Reshoot_06_single_1080x1620.mp4')
vd = OrderedDict()
i = 0

print('loading mp4')
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret is True:
        vd[i] = frame
    else:
        break
    i += + 1
print(len(vd), 'frames loaded')


for i in range(200):

    # Image Resize for dev
    # outframe = cv.resize(img, None, fx=0.4, fy=0.4)

    img = vd[i]
    cv2.imshow('frame', img)
    key = cv2.waitKey(20)

