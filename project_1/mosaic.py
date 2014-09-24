#!/usr/bin/env python
import cv2
import numpy as np
import pano_stitcher

image_1 = cv2.imread("my_pano_pics/gdc_1.png")
image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2BGRA)
image_2 = cv2.imread("my_pano_pics/gdc_2.png")
image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2BGRA)
image_3 = cv2.imread("my_pano_pics/gdc_3.png")
image_3 = cv2.cvtColor(image_3, cv2.COLOR_BGR2BGRA)

images = []
origins = []

homography = pano_stitcher.homography(image_2, image_1)
image_1_warped, origin = pano_stitcher.warp_image(image_1, homography)
images.append(image_1_warped)
origins.append(origin)

images.append(image_2)
origins.append((0, 0))

homography = pano_stitcher.homography(image_2, image_3)
print homography
image_3_warped, origin = pano_stitcher.warp_image(image_3, homography)
images.append(image_3_warped)
origins.append(origin)

print origins

panorama = pano_stitcher.create_mosaic(images, origins)

cv2.imwrite("gdc_pano.png", panorama)
# cv2.imshow("panorama", panorama)
# cv2.waitKey(0)
