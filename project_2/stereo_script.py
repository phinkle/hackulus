import cv2
import numpy as np
import StringIO
import stereo as stereo
import argparse

parser = argparse.ArgumentParser(description='TEST DESCRIPTION')
parser.add_argument('-l', '--left', help='Input left image', required=True)
parser.add_argument('-r', '--right', help='Input right image', required=True)
args = parser.parse_args()

print ("Left image: %s" % args.left)
print ("Right image: %s" % args.right)

image_left = cv2.imread(args.left)
image_right = cv2.imread(args.right)

focal_length = 10

F, h_left, h_right = stereo.rectify_pair(image_left, image_right)

r_image_left = cv2.warpPerspective(image_left, h_left,
                                   image_left.shape[:2])

r_image_right = cv2.warpPerspective(image_right, h_right,
                                    image_right.shape[:2])

disp = stereo.disparity_map(image_left, image_right)

ply_string = stereo.point_cloud(disp, image_left, focal_length)

cv2.imwrite("image_left.jpg", r_image_left)
cv2.imwrite("image_right.jpg", r_image_right)
cv2.imwrite("disparity.jpg", disp)

with open("out.ply", 'w') as f:
    f.write(ply_string)
