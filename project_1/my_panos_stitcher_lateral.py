""" PANORAMA STITCHER
        Python script to create our personal panorama
"""

import cv2
import numpy as np
import pano_stitcher as ps

img1 = cv2.imread('my_panos/gdc1.jpg')
img2 = cv2.imread('my_panos/gdc2.jpg')
img3 = cv2.imread('my_panos/gdc3.jpg')
img4 = cv2.imread('my_panos/gdc4.jpg')
img5 = cv2.imread('my_panos/gdc5.jpg')
img6 = cv2.imread('my_panos/gdc6.jpg')

homography1 = ps.homography(img4, img1)
homography2 = ps.homography(img4, img2)
homography3 = ps.homography(img4, img3)
homography5 = ps.homography(img4, img5)
homography6 = ps.homography(img4, img6)

warped1, tuple1 = ps.warp_image(img1, homography1)
warped2, tuple2 = ps.warp_image(img2, homography2)
warped3, tuple3 = ps.warp_image(img3, homography3)
warped5, tuple5 = ps.warp_image(img5, homography5)
warped6, tuple6 = ps.warp_image(img6, homography6)

images = (img1, img2, img3, img4, img5, img6)
origins = (tuple1, tuple2, tuple3, (0, 0), tuple5, tuple6)

dst = ps.create_mosaic(images, origins)

cv2.imwrite("my_panos/panorama_lateral.png", dst)
