"""Project 3: Tracking.

In this project, you'll track objects in videos.
"""

import cv2
import math
import numpy as np


def _trackCircle(video, dp, par1, par2, minDistance, showTests=False):
    """Function to detect circle every frame, we used trackCam instead
       It takes in a few parameters of HoughCircles
       showTests activates imshow to preview results
    """

    bounds = list()
    success, image = video.read()

    while success:
        img = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)
        img = cv2.medianBlur(img, 5)

        circlesFound = cv2.HoughCircles(
                img,
                cv2.cv.CV_HOUGH_GRADIENT,
                dp,
                minDistance,
                param1=par1,
                param2=par2
        )

        # if circle is detected...
        if circlesFound is not None and len(circlesFound) >= 1:
            circlesFound = np.uint16(np.around(circlesFound))
            # takes our _one_ circle and gets rectangular bounds
            j = circlesFound[0, :]
            minx, miny, maxx, maxy = (
                    j[0] - j[2],
                    j[1] - j[2],
                    j[0] + j[2],
                    j[1] + j[2]
            )
            bounds.append(minx, miny, maxx, maxy)
            # draws rectangular visual for showTests
            cv2.rectangle(image, (minx, miny), (maxx, maxy), (255, 0, 0), 2)
        else:
            raise RuntimeError("detected circles less than 1")

        if showTests:
            print bounds[i]
            cv2.imshow("TestPic", image)
            cv2.waitKey(0)

        success, image = video.read()

    return bounds


def _trackCam(video, dp, par1, par2, minDistance, showTests=False, alt=False):
    """Function to detect circle via meanshifting
       It takes in a few parameters of HoughCircles
       showTests activates imshow to preview results
    """

    # cam code pulled from online example
    bounds = list()

    success, image = video.read()

    img = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)
    if not alt:
        img = cv2.medianBlur(img, 5)

    if showTests:
        cv2.imshow("TestImage", image)
        cv2.waitKey(0)

    circlesFound = cv2.HoughCircles(
                img,
                cv2.cv.CV_HOUGH_GRADIENT,
                dp,
                minDistance,
                param1=par1,
                param2=par2
    )

    # track window coordinates, minx, miny, and width and height
    x1, y1, side = 0, 0, 0

    if circlesFound is not None and len(circlesFound) >= 1:
        j = circlesFound[0, 0]
        x1, y1, side = j[0] - j[2], j[1] - j[2], j[2]
    else:
        raise RuntimeError("no circles found")

    # tracking window modifications, if alt is used, make window larger
    if alt:
        side = int(3.6 * side)
    else:
        side = int(2 * side)
    track_window = (x1, y1, side, side)

    # set up the ROI for tracking
    roi = image[x1:x1 + side, y1:y1 + side]

    if showTests:
        print track_window
        cv2.imshow("TestRangeofIndex", roi)

    # sets up circumstances for using calcBackProject and camShift
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
                hsv_roi,
                np.array((0., 60., 32.)),
                np.array((180., 255., 255.))
    )
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # Setup the termination criteria for meanShift
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1)

    while success:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 3)

        # apply meanshift to get the new location
        success, track_window = cv2.meanShift(dst, track_window, term_crit)

        x, y, w, h = track_window
        bounds.append((x, y, x + w, y + h))

        if showTests:
            print track_window
            print bounds
            cv2.rectangle(image, (x, y), (x + w, y + h), 255, 2)
            cv2.imshow("TestPic", image)
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
    return _trackCam(video, 1, 45, 10, 1000, alt=True)


def track_ball_4(video):
    """As track_ball_1, but for ball_2.mov."""
    return _trackCam(video, 1.6, 45, 10, 1000)


def _getParam():
    """Generator for track_face, gets parameters for face_cascade"""
    for i in range(1, 10):
        for j in range(2, 10):
            yield i, j
    raise RuntimeError("No faces found, implied")


def track_face(video):
    """As track_ball_1, but for face.mov.
       uses a Cascade Classifier
    """
    bounds = list()

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    success, image = video.read()

    while success:
        # this function will detect a face every frame and assumes 1 face
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = cv2.medianBlur(img, 5)

        facesFound = face_cascade.detectMultiScale(img, 1.2, 6)

        # under the circumstance that we don't find a face
        if facesFound is None or len(facesFound) != 1:
            # use expensive check to get usable parameters when default fails
            # _getParam is used to generate parameters for face_cascade
            for i, j in _getParam():
                facesFound = face_cascade.detectMultiScale(
                                img,
                                (1.0 + 0.1 * i),
                                j
                            )
                if not (facesFound is None or len(facesFound) != 1):
                    break

        # get the single face and put it into bounds
        x, y, w, h = facesFound[0]
        bounds.append((x, y, x + w, y + h))

        success, image = video.read()

    return bounds
