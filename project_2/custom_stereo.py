import cv2
import numpy as np
import StringIO
import stereo as stereo

#image_left = cv2.imread('test_data/kitchen_left.jpg')
#image_right = cv2.imread('test_data/kitchen_right.jpg')

#image_left = cv2.imread('img_left.jpg')
#image_right = cv2.imread('img_right.jpg')
image_left = cv2.imread('test_data/tsukuba/left.png')
image_right = cv2.imread('test_data/tsukuba/right.png')

print(image_left.shape)
print(image_right.shape)

focal_length = 10

F, h_left, h_right = stereo.rectify_pair(image_left, image_right)

l_height, l_width, l_depth = image_left.shape
left_shape = (l_width, l_height)
rectify_left, _ = stereo.warp_image(image_left, h_left)

r_height, r_width, r_depth = image_right.shape
right_shape = (r_width, r_height)
rectify_right, _ = stereo.warp_image(image_right, h_right)

disparity = stereo.disparity_map(rectify_left, rectify_right)

"""ply_string = stereo.point_cloud(disparity, image_left, focal_length)

with open("out.ply", 'w') as f:
    f.write(ply_string)

print(rectify_left)
cv2.imwrite("image_left.jpg", rectify_left)
cv2.imwrite("image_right.jpg", rectify_right)
"""
