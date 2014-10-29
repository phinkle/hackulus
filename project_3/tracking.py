"""Project 3: Tracking.

In this project, you'll track objects in videos.
"""

import cv2
import math
import numpy as np

def track_ball(video):
    bounds = []
    success, image = video.read()

    while success:
      image = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY, dstCn=1)
      # image = cv2.blur(image, (5,5), 0)
      bounds.append(cv2.HoughCircles(image, cv2.cv.CV_HOUGH_GRADIENT, 1.2, 100))
      success, image = video.read()

    print bounds
  
def track_ball_1(video):
    """Track the ball's center in 'video'.

    Arguments:
      video: an open cv2.VideoCapture object containing a video of a ball
        to be tracked.

    Outputs:
      a list of (min_x, min_y, max_x, max_y) four-tuples containing the pixel
      coordinates of the rectangular bounding box of the ball in each frame.
    """
    track_ball(video)
    pass


def track_ball_2(video):
    """As track_ball_1, but for ball_2.mov."""
    # track_ball(video)
    pass


def track_ball_3(video):
    """As track_ball_1, but for ball_2.mov."""
    # track_ball(video)
    pass


def track_ball_4(video):
    """As track_ball_1, but for ball_2.mov."""
    # track_ball(video)
    pass


def track_face(video):
    pass
