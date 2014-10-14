""" PANORAMA STITCHER
        Python script to create our personal panorama
"""

import cv2
import numpy as np
import pano_stitcher as ps


img_left = cv2.imread('my_panos/wics1.jpg')
img_left_bw = cv2.imread('my_panos/wics1.jpg', 0)
img_center = cv2.imread('my_panos/wics2.jpg')
img_right = cv2.imread('my_panos/wics3.jpg')
img_center_bw = cv2.imread('my_panos/wics2.jpg', 0)
img_right_bw = cv2.imread('my_panos/wics3.jpg', 0)

homography1 = ps.homography(img_center_bw, img_left_bw)
homography2 = ps.homography(img_center_bw, img_right_bw)
homography1 = ps.homography(img_center_bw, img_left_bw)
homography2 = ps.homography(img_center_bw, img_right_bw)

warped_left, tuple_left = ps.warp_image(img_left, homography1)
warped_right, tuple_right = ps.warp_image(img_right, homography2)

images = (warped_left, warped_right, img_center)
origins = (tuple_left, tuple_right, (0, 0))

dst = ps.create_mosaic(images, origins)

cv2.imwrite("my_panos/panorama.png", dst)
