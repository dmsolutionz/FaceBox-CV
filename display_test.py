import cv2
from collections import OrderedDict

# Create VideoCapture object from input file
cap = cv2.VideoCapture('data/JC_C_Reshoot_06_single_1080x1620.mp4')
vd = OrderedDict()
i = 0


print('Loading mp4')
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret is True:
        vd[i] = frame
    else:
        break
    i += + 1

print(len(vd), 'frames loaded')

# ------------------------------------------------ #
#                 Write to MP4
# ------------------------------------------------ #

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
print('\nOutput', frame_width, 'x', frame_height)

out = cv2.VideoWriter('data/output.mp4',
                      0x7634706d,
                      24.0,
                      (frame_width, 840))

# Write the frame into the file 'output.avi'
for k in range(i):
    # Crop tall video
    # ---------------------------------- #
    y1, y2 = 780, 1620
    x1, x2 = 0, 1080
    outframe = vd[k][y1:y2, x1:x2].copy()
    # cv2.imshow('frame', outframe)
    # key = cv2.waitKey(3000)
    # ---------------------------------- #
    out.write(outframe)

print('\t', len(vd), 'frames')
# release video and write objects
out.release()
cap.release()
