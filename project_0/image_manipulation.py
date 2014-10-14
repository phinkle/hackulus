"""Project 0: Image Manipulation with OpenCV.

In this assignment, you will implement a few basic image
manipulation tasks using the OpenCV library.

Use the unit tests is image_manipulation_test.py to guide
your implementation, adding functions as needed until all
unit tests pass.
"""
import cv2
import numpy as np
# TODO: Implement!

<<<<<<< HEAD
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
=======

def flip_image(imageFile, horizontal, vertical):
    if (horizontal and vertical):
        return cv2.flip(imageFile, -1)
    elif(vertical):
        return cv2.flip(imageFile, 0)
    elif(horizontal):
        return cv2.flip(imageFile, 1)
    else:
        return imageFile


def negate_image(imageFile):
    return cv2.subtract(np.full(imageFile.shape, 255, dtype=np.uint8),
                        imageFile)


def swap_blue_and_green(imageFile):
    b, g, r = cv2.split(imageFile)
    return cv2.merge([g, b, r])
>>>>>>> d7132e25c1b67417913bfcf00799a29036f2bb39
