"""Project 2: Stereo vision.

In this project, you'll extract dense 3D information from stereo image pairs.
"""

import cv2
import math
import numpy as np
import StringIO


def rectify_pair(image_left, image_right, viz=False):
    """Computes the pair's fundamental matrix and rectifying homographies.

    Arguments:
      image_left, image_right: 3-channel images making up a stereo pair.

    Returns:
      F: the fundamental matrix relating epipolar geometry between the pair.
      H_left, H_right: homographies that warp the left and right image so
        their epipolar lines are corresponding rows.
    """

    # Initiate SIFT detector
    sift = cv2.SIFT()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(image_left, None)
    kp2, des2 = sift.detectAndCompute(image_right, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])

    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.LMEDS)
    src_pts = src_pts.flatten()
    dst_pts = dst_pts.flatten()

    retval, H_left, H_right = cv2.stereoRectifyUncalibrated(
        src_pts, dst_pts, F, image_left.shape[:2])

    print("HLEFT")
    print(H_left)

    return F, H_left, H_right


def disparity_map(image_left, image_right):
    """Compute the disparity images for image_left and image_right.

    Arguments:
      image_left, image_right: rectified stereo image pair.

    Returns:
      an single-channel image containing disparities in pixels,
        with respect to image_left's input pixels.
    """

    window_size = 3
    min_disp = 16
    num_disp = 112 - min_disp
    stereo = cv2.StereoSGBM(minDisparity=min_disp,
                            numDisparities=num_disp,
                            SADWindowSize=window_size,
                            uniquenessRatio=10,
                            speckleWindowSize=100,
                            speckleRange=32,
                            disp12MaxDiff=1,
                            P1=8 * 3 * (window_size ** 2),
                            P2=32 * 3 * (window_size ** 2),
                            fullDP=False
                            )

    temp_disp = stereo.compute(image_left, image_right) / 16.0
    disp = np.array(temp_disp, dtype="uint8")

    return disp

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''


def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    # with open(fn, 'w') as f:
    fn.write(ply_header % dict(vert_num=len(verts)))
    np.savetxt(fn, verts, '%f %f %f %d %d %d')
    return fn


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


def point_cloud(disparity_image, image_left, focal_length):
    """Create a point cloud from a disparity image and a focal length.

    Arguments:
      disparity_image: disparities in pixels.
      image_left: BGR-format left stereo image, to color the points.
      focal_length: the focal length of the stereo camera, in pixels.

    Returns:
      A string containing a PLY point cloud of the 3D locations of the
        pixels, with colors sampled from left_image. You may filter low-
        disparity pixels or noise pixels if you choose.
    """
    h, w = image_left.shape[:2]
    Q = np.float32([[1, 0,  0, w / 2],
                    [0, -1,  0,  h / 2],  # turn points 180 deg around x-axis,
                    [0, 0, focal_length,  0],  # so that y-axis looks up
                    [0, 0,  0,  1]])
    points = cv2.reprojectImageTo3D(disparity_image, Q)
    colors = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)
    mask = disparity_image > disparity_image.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'out.ply'
    cloud = StringIO.StringIO()
    # items = write_ply(out_fn, out_points, out_colors)
    cloudStuff = write_ply(cloud, out_points, out_colors)
    print cloudStuff.getvalue()
    return cloudStuff.getvalue()
