"""Assignment 1: Image Manipulation with OpenCV.

In this assignment, you will implement a few basic image
manipulation tasks using the OpenCV library.

Use the unit tests is image_manipulation_test.py to guide
your implementation, adding functions as needed until all
unit tests pass.
"""

# TODO: Implement!

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
    out = []
    for row in img:
        out.append(cv2.absdiff(255, row))
    return numpy.asarray(out)


def swap_red_and_green(img):
    return cv2.merge([cv2.split(img)[1], cv2.split(img)[0], cv2.split(img)[2]])
