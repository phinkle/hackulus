"""Project 0: Image Manipulation with OpenCV.

In this assignment, you will implement a few basic image
manipulation tasks using the OpenCV library.

Use the unit tests is image_manipulation_test.py to guide
your implementation, adding functions as needed until all
unit tests pass.
"""

import cv2
import pep8
import numpy


def flip_image(img, horizontal, vertical):
    flip_code = None

    if horizontal and not vertical:
        flip_code = 1

    elif vertical and not horizontal:
        flip_code = 0

    elif horizontal and vertical:
        flip_code = -1

    else:
        return img

    return cv2.flip(img, flip_code)


def negate_image(img):
    return cv2.bitwise_not(img)

def swap_blue_and_green(img):
    dst = [img.copy()]
    cv2.mixChannels([img], dst, numpy.array([1, 0, 0, 1, 2, 2]))
    return dst[0]
