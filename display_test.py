import cv2
from collections import OrderedDict

# Create VideoCapture object from input file
cap = cv2.VideoCapture('data/JC_C_Reshoot_06_single_1080x1620.mp4')
vd = OrderedDict()
i = 0

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret is True:
        vd[i] = frame
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(100)
    else:
        break
    i += + 1
print(len(vd), 'frames loaded')


# for img in vd:

#     # Image Resize for dev
#     # outframe = cv.resize(img, None, fx=0.4, fy=0.4)

#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     cv2.imshow('frame', img)

#     key = cv2.waitKey(100)  # pause for 100 ms
#     if key == 27:  # if ESC is pressed, exit loop
#         cv2.destroyAllWindows()
#         break