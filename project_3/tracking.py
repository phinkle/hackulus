"""Project 3: Tracking.

In this project, you'll track objects in videos.
"""

import cv2
import math
import numpy as np


def _trackCircle(video, dp, par1, par2, minD, showTests=False, alt=False):
    """Function to detect circle every frame, we used trackCam instead"""

    bounds = list()

    success, image = video.read()
    i = 0
    while success:
        img = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)
        img = cv2.medianBlur(img, 5)

        circlesFound = cv2.HoughCircles(img, cv2.cv.CV_HOUGH_GRADIENT,
                        dp, minD, param1=par1, param2=par2)

        if circlesFound is not None and len(circlesFound) >= 1:
            circlesFound = np.uint16(np.around(circlesFound))
            j = circlesFound[0, :]
            bounds.append((j[0] - j[2], j[1] - j[2], j[0] + j[2], j[1] + j[2]))
            cv2.rectangle(image, (j[0] - j[2], j[1] - j[2]),
                        (j[0] + j[2], j[1] + j[2]), (255, 0, 0), 2)
        else:
            raise RuntimeError("detected circles less than 1")

        if showTests and i < 30:
            print bounds[i]
            cv2.imshow("alpha" + repr(i) + ".png", image)
            i += 1
        success, image = video.read()

    return bounds


def _findBackground(video):
    """Function to detect background, incomplete"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgbg = cv2.BackgroundSubtractorMOG()
    success, image = video.read()
    fgmask2 = None
    while success:
        fgmask = fgbg.apply(image)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        cv2.imshow('frame', fgmask)
        other = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)
        cv2.imshow('frame2', other)
        other2 = cv2.subtract((255 - other), (255 - fgmask))
        cv2.imshow('frame3', other2)
        cv2.waitKey(0)
        print cv2.HoughCircles(other, cv2.cv.CV_HOUGH_GRADIENT,
                        1.6, 5000, param1=45, param2=10)
        fgmask2 = fgmask
        success, image = video.read()
    return bg


def _trackCam(video, dp, par1, par2, minD, showTests=False, alt=False):
    """Function to detect circle via meanshifting"""

    # cam code pulled from online example
    bounds = list()

    success, image = video.read()
    success2 = success
    img = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)
    img = cv2.medianBlur(img, 5)
    use = img  # use this img, unless alt is used

    # if alt is true, then apply alternative detection (for track ball 3)
    if showTests:
        cv2.imshow("aa", image)
        cv2.waitKey(99999)

    circles = cv2.HoughCircles(use, cv2.cv.CV_HOUGH_GRADIENT, dp, minD,
                        param1=par1, param2=par2)

    x1, y1, w1, h1 = 0, 0, 0, 0

    if circles is not None and len(circles) >= 1:
        j = circles[0, 0]
        x1, y1, w1, h1 = j[0] - j[2], j[1] - j[2], j[2], j[2]
    else:
        raise RuntimeError("no circles found")

    track_window = (x1, y1, w1 + w1, h1 + h1)

    if showTests:
        print track_window

    # set up the ROI for tracking
    roi = image[x1:x1 + w1 + w1, y1:y1 + h1 + h1]

    if showTests:
        cv2.imshow("ok", roi)

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)),
                np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # Setup the termination criteria
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while success and success2:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = cv2.medianBlur(hsv, 5)

        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # apply meanshift to get the new location
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        x, y, w, h = track_window
        bounds.append((x, y, x + w, y + h))

        if showTests:
            print track_window
            print bounds
            cv2.rectangle(image, (x, y), (x + w, y + h), 255, 2)
            cv2.imshow("alpha.png", image)
            cv2.waitKey(0)

        success, image = video.read()

    cv2.destroyAllWindows()
    return bounds


def track_ball_1(video):
    """Track the ball's center in 'video'.

    Arguments:
        video: an open cv2.VideoCapture object containing a video of a ball
        to be tracked.

    Outputs:
        a list of (min_x, min_y, max_x, max_y) four-tuples containing the pixel
        coordinates of the rectangular bounding box of the ball in each frame.
    """
    return _trackCam(video, 1.95, 50, 20, 30)


def track_ball_2(video):
    """As track_ball_1, but for ball_2.mov."""
    return _trackCam(video, 1.6, 45, 10, 1000)


def track_ball_3(video):
    """As track_ball_1, but for ball_2.mov."""
    #_findBackground(video)
    return _trackCircle(video, 1.95, 45, 10, 1000, alt=True)


def track_ball_4(video):
    """As track_ball_1, but for ball_2.mov."""
    return _trackCam(video, 1.6, 45, 10, 1000)


def _getParam():
    """Generator for track_face, gets parameters"""
    for i in range(1, 10):
        for j in range(2, 10):
            yield i, j


def track_face(video):
    """As track_ball_1, but for face.mov."""
    return
    bounds = list()

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    success, image = video.read()

    while success:
        # this function will detect a face every frame and assumes 1 face
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = cv2.medianBlur(img, 5)

        facesFound = face_cascade.detectMultiScale(img, 1.2, 6)

        if facesFound is None or len(facesFound) != 1:
            # use expensive check to get usable parameters when default fails
            for i, j in _getParam():
                facesFound = face_cascade.detectMultiScale(img,
                                (1.0 + 0.1 * i), j)
                if not (facesFound is None or len(facesFound) != 1):
                    break

        x, y, w, h = facesFound[0]
        bounds.append((x, y, x + w, y + h))

        success, image = video.read()

    return bounds
