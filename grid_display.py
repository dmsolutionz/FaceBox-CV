import cv2
import numpy as np


def grdifiy_four(image):
    """
    Function takes an OpenCV image and returns a grid with B&W pattern

    Args:
        Opencv image object

    image = cv2.imread('data/lfw_2018.JPG')
    """
    # Add to bw and make the grey scale image have three channels
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grey = cv2.flip(grey, 1)
    grey_3_channel = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

    hstack_1 = np.hstack((image, grey_3_channel))
    hstack_2 = np.hstack((grey_3_channel, image))

    four_up = np.vstack((hstack_1, hstack_2))

    return four_up


if __name__ == "__main__":
    image = cv2.imread('data/lfw_2018.JPG')
    image = cv2.resize(image, (0, 0), None, .5, .5)

    four_up = grdifiy_four(image)

    cv2.imshow('4-up Grid', four_up)
    cv2.waitKey()
