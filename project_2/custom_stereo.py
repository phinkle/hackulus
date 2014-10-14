import cv2
import numpy as np
import StringIO
import stereo as stereo

"""Tikki Image"""
# image_left = cv2.imread('test_data/cones/im2.png')
# image_right = cv2.imread('test_data/cones/im6.png')

# image_left = cv2.pyrDown(cv2.pyrDown(cv2.imread('otherData/l5.jpg')))
# image_right = cv2.pyrDown(cv2.pyrDown(cv2.imread('otherData/r6.jpg')))

image_left = cv2.imread('test_data/tsukuba/left.png')
image_right = cv2.imread('test_data/tsukuba/right.png')
# image_left = cv2.imread('left1.png')
# image_right = cv2.imread('right1.png')

#r_image_left = cv2.imread('test_data/rectified_left.png')
#r_image_right = cv2.imread('test_data/rectified_right.png')

focal_length = 10

print(image_left.shape)
print(image_right.shape)

F, h_left, h_right = stereo.rectify_pair(image_left, image_right)

l_height, l_width, l_depth = image_left.shape
left_shape = (l_width, l_height)
# rectify_left, something = stereo.warp_image(image_left, h_left)
rectify_left = cv2.warpPerspective(image_left, h_left, left_shape)

r_height, r_width, r_depth = image_right.shape
right_shape = (r_width, r_height)
# rectify_right, something2 = stereo.warp_image(image_right, h_right)
rectify_right = cv2.warpPerspective(image_right, h_right, right_shape)

#disparity = stereo.disparity_map(rectify_left, rectify_right)
disparity = stereo.disparity_map(image_left, image_right)

ply_string = stereo.point_cloud(disparity, image_left, focal_length)

cv2.imwrite("image_left.jpg", rectify_left)
cv2.imwrite("image_right.jpg", rectify_right)
cv2.imwrite("disparity.jpg", disparity)

with open("out.ply", 'w') as f:
    f.write(ply_string)
