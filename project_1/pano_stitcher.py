"""Project 1: Panorama stitching.

In this project, you'll stitch together images to form a panorama.

A shell of starter functions that already have tests is listed below.

TODO: Implement!
"""

import cv2
<<<<<<< HEAD
import numpy as np
=======
import cv
import numpy as np

MIN_MATCH_COUNT = 10
>>>>>>> d7132e25c1b67417913bfcf00799a29036f2bb39


def homography(image_a, image_b):
    """Returns the homography mapping image_b into alignment with image_a.

    Arguments:
      image_a: A grayscale input image.
      image_b: A second input image that overlaps with image_a.

    Returns: the 3x3 perspective transformation matrix (aka homography)
             mapping points in image_b to corresponding points in image_a.
    """

<<<<<<< HEAD
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

    dst_pts = np.float32([kp_a[m.queryIdx].pt for m in good])
    src_pts = np.float32([kp_b[m.trainIdx].pt for m in good])

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return M
=======
    gray_a = np.copy(image_a)
    gray_b = np.copy(image_b)

    # Generate keypoints and descriptors from SIFT
    sift = cv2.SIFT()

    kp_a, des_a = sift.detectAndCompute(gray_a, None)
    kp_b, des_b = sift.detectAndCompute(gray_b, None)

    # Apply matcher to find best interest points
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des_a, des_b, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance/n.distance < 0.9:
            good.append(m)

    # Find homography
    if len(good) > MIN_MATCH_COUNT:
        dst = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        src = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)

        return M

    pass
>>>>>>> d7132e25c1b67417913bfcf00799a29036f2bb39


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

<<<<<<< HEAD
    top_left = np.array([0, 0, 1])
    bottom_left = np.array([image.shape[0], 0, 1])
    top_right = np.array([0, image.shape[1], 1])
    bottom_right = np.array([image.shape[0], image.shape[1], 1])

    origin = (int(homography[0][2]), int(homography[1][2]))

    homography[0][2] = 0
    homography[1][2] = 0

    top_left_warped = np.dot(homography, top_left)
    bottom_left_warped = np.dot(homography, bottom_left)
    top_right_warped = np.dot(homography, top_right)
    bottom_right_warped = np.dot(homography, bottom_right)

    top_left = top_left_warped / top_left_warped[2]
    bottom_left = bottom_left_warped / bottom_left_warped[2]
    top_right = top_right_warped / top_right_warped[2]
    bottom_right = bottom_right_warped / bottom_right_warped[2]

    new_width = max(top_left[1], bottom_left[1], top_right[1], bottom_right[1])
    new_height = max(top_left[0], bottom_left[0], top_right[0],
                     bottom_right[0])

    new_size = (int(new_width), int(new_height))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    warped = cv2.warpPerspective(image, homography, new_size)

    return (warped, origin)
=======
    # Find the four corners dotted with the homography
    corner_lu = np.dot(homography, (0, 0, 1))
    corner_ru = np.dot(homography, (0, image.shape[0], 1))
    corner_ld = np.dot(homography, (image.shape[1], 0, 1))
    corner_rd = np.dot(homography, (image.shape[1], image.shape[0], 1))

    # Find the minimum and maximum, the origin would be the minimum x and y
    # and homogenize the points
    origin_x = int(min(corner_lu[0]/corner_lu[2], corner_ru[0]/corner_ru[2],
                       corner_ld[0]/corner_ld[2], corner_rd[0]/corner_rd[2]))
    origin_y = int(min(corner_lu[1]/corner_lu[2], corner_ru[1]/corner_ru[2],
                       corner_ld[1]/corner_ld[2], corner_rd[1]/corner_rd[2]))
    max_x = int(max(corner_lu[0]/corner_lu[2], corner_ru[0]/corner_ru[2],
                    corner_ld[0]/corner_ld[2], corner_rd[0]/corner_rd[2]))
    max_y = int(max(corner_lu[1]/corner_lu[2], corner_ru[1]/corner_ru[2],
                    corner_ld[1]/corner_ld[2], corner_rd[1]/corner_rd[2]))

    # Multiply the homography by the offset to correctly translate it
    offset_matrix = [[1, 0, -origin_x], [0, 1, -origin_y], [0, 0, 1]]
    new_M = np.dot(offset_matrix, homography)

    width = int(max_x - origin_x)
    height = int(max_y - origin_y)

    warped_image = cv2.warpPerspective(image, new_M, (width, height))

    return warped_image, (origin_x, origin_y)
>>>>>>> d7132e25c1b67417913bfcf00799a29036f2bb39


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
<<<<<<< HEAD

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
                pano_y = y + new_origins[i][1]
                pano_x = x + new_origins[i][0]

                panorama[pano_y, pano_x] = image[y, x]

    return panorama
=======
    # Find the x values and y values of the origin points for the size
    x_vals = {point[0] for point in origins}
    y_vals = {point[1] for point in origins}

    x_min = min(x_vals)
    y_min = min(y_vals)
    x_max = max(x_vals)
    y_max = max(y_vals)

    sizes = {im.shape for im in images}
    heights = {sh[0] for sh in sizes}
    widths = {sh[1] for sh in sizes}

    height_max = max(heights)
    width_max = max(widths)

    overall_height = abs(y_min) + y_max + height_max
    overall_width = abs(x_min) + x_max + width_max

    scaled_images = []

    # Translate the images to their appropriate place in the larger image
    for img, origin in zip(images, origins):
        M = np.float32([[1, 0, origin[0]+abs(x_min)],
                        [0, 1, origin[1]+abs(y_min)]])
        scaled_images.append(cv2.warpAffine(img, M, (overall_width,
                                                     overall_height)))

    dst = scaled_images[0]

    # Copy images over to the larger image with all other images
    for img in scaled_images:
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for k in range(img.shape[2]):
                    if (img[i, j, k] != 0):
                        # Commented code allows for blending between overlap
                        # if(dst[i, j, k] != 0):
                        #    dst[i, j, k] = (dst[i, j, k]/2 + img[i, j, k]/2)
                        # else:
                        dst[i, j, k] = img[i, j, k]

    return dst


def composite_image(start_row, start_col, img_a, img_b):
    """Combine a smaller image into a larger image.

        Arguments:
        start_row: row that you want the smaller image to start at
                   in the larger image.
        start_col: col that you want the smaller image to start at
                   in the larger image.
        img_a: the larger image, as the destination for the composite.
        img_b: the smaller image that will be composited in img_a.

        Returns: img_a that has had img_b composited on top of it.
        """
    end_row = img_b.shape[1] + start_row
    end_col = img_b.shape[0] + start_col

    # Numpy operation to composite the image
    img_a[start_col:end_col, start_row:end_row] = img_b
    return img_a
>>>>>>> d7132e25c1b67417913bfcf00799a29036f2bb39
