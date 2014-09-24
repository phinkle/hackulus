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
