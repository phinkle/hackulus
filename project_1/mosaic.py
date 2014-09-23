#!/usr/bin/env python
import cv2
import numpy as np
import pano_stitcher


books_1 = cv2.imread("test_data/books_1.png")
books_2 = cv2.imread("test_data/books_2.png")
books_3 = cv2.imread("test_data/books_3.png")

images = []
origins = []

homography = pano_stitcher.homography(books_2, books_1)
books_1_warped, origin = pano_stitcher.warp_image(books_1, homography)
images.append(books_1_warped)
origins.append(origin)

images.append(books_2)
origins.append((0, 0))

homography = pano_stitcher.homography(books_3, books_2)
books_3_warped, origin = pano_stitcher.warp_image(books_3, homography)
images.append(books_3_warped)
origins.append(origin)

print origins

panorama = pano_stitcher.create_mosaic(images, origins)

cv2.imshow("panorama", panorama)
cv2.waitKey(0)
