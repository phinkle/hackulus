#!/usr/bin/env python
import cv2
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(".."))

import pano_stitcher

if len(sys.argv) < 4:
    print "Usage: {} image1 image2 image3".format(sys.argv[0])
    sys.exit()

image_1 = cv2.imread(sys.argv[1])
image_2 = cv2.imread(sys.argv[2])
image_3 = cv2.imread(sys.argv[3])

images = []
origins = []

homography = pano_stitcher.homography(image_2, image_1)
image_1_warped, origin = pano_stitcher.warp_image(image_1, homography)
images.append(image_1_warped)
origins.append(origin)

image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2BGRA)
images.append(image_2)
origins.append((0, 0))

homography = pano_stitcher.homography(image_2, image_3)
image_3_warped, origin = pano_stitcher.warp_image(image_3, homography)
images.append(image_3_warped)
origins.append(origin)

panorama = pano_stitcher.create_mosaic(images, origins)

cv2.imwrite("panorama.png", panorama)
