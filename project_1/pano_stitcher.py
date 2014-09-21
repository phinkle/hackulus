"""Project 1: Panorama stitching.

In this project, you'll stitch together images to form a panorama.

A shell of starter functions that already have tests is listed below.

TODO: Implement!
"""

import cv2
import numpy as np


def homography(image_a, image_b):
    """Returns the homography mapping image_b into alignment with image_a.

    Arguments:
      image_a: A grayscale input image.
      image_b: A second input image that overlaps with image_a.

    Returns: the 3x3 perspective transformation matrix (aka homography)
             mapping points in image_b to corresponding points in image_a.
    """

    sift = cv2.SIFT()
    kp_a, des_a = sift.detectAndCompute(image_a, None)
    kp_b, des_b = sift.detectAndCompute(image_b, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    bf = cv2.BFMatcher()
    matches = flann.knnMatch(des_a, des_b, k=2)

    good = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    # print len(good)
    dst_pts = np.float32([kp_a[m.queryIdx].pt for m in good])
    src_pts = np.float32([kp_b[m.trainIdx].pt for m in good])

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return M


def warp_image(image, homography):
    """Warps 'image' by 'homography'

    Arguments:
      image: a 3-channel image to be warped.
      homography: a 3x3 perspective projection matrix mapping points
                  in the frame of 'image' to a target frame.

    Returns:
      - a new 4-channel image containing the warped input, resized to contain
        the new image's bounds. Translation is offset so the image fits exactly
        within the bounds of the image. The fourth channel is an alpha channel
        which is zero anywhere that the warped input image does not map in the
        output, i.e. empty pixels.
      - an (x, y) tuple containing location of the warped image's upper-left
        corner in the target space of 'homography', which accounts for any
        offset translation component of the homography.
    """
    new_size = (int(homography[1][1] * image.shape[1]),
                int(homography[0][0] * image.shape[0]))

    image_t = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    warped = cv2.warpPerspective(image_t, homography, new_size)
    origin = (int(homography[0][2]), int(homography[1][2]))

    cv2.imwrite("warped.png", warped)

    return (warped, origin)


def create_mosaic(images, origins):
    """Combine multiple images into a mosaic.

    Arguments:
      images: a list of 4-channel images to combine in the mosaic.
      origins: a list of the locations upper-left corner of each image in
               a common frame, e.g. the frame of a central image.

    Returns: a new 4-channel mosaic combining all of the input images. pixels
             in the mosaic not covered by any input image should have their
             alpha channel set to zero.
    """

    top_left = (min(x for x, y in origins), min(y for x, y in origins))

    new_origins = [(x-top_left[0], y-top_left[1]) for x, y in origins]

    pano_width = 0
    pano_height = 0

    for i in range(len(images)):
        image = images[i]
        x, y = new_origins[i]
        pano_width = max(pano_width, image.shape[1]+x)
        pano_height = max(pano_height, image.shape[0]+y)

    panorama = np.zeros((pano_height, pano_width, 4), np.uint8)

    for i in range(len(images)):
        image = images[i]

        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                panorama[y+new_origins[i][1], x+new_origins[i][0]] = image[y, x]

    return panorama
