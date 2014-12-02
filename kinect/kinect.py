import cv2
import os, sys
import numpy as np
from Queue import Queue


capture = cv2.VideoCapture(cv2.cv.CV_CAP_OPENNI)
capture.set(cv2.cv.CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, cv2.cv.CV_CAP_OPENNI_VGA_30HZ)

print capture.get(cv2.cv.CV_CAP_PROP_OPENNI_REGISTRATION)
num_frame = -1
num_taken = 0
depth_range = 750

while True:
    if not capture.grab():
        print "Unable to Grab Frames from camera"
        break
    okay1, depth_map = capture.retrieve(0, cv2.cv.CV_CAP_OPENNI_DEPTH_MAP)
    if not okay1:
        print "Unable to Retrieve Disparity Map from camera"
        break

    okay2, bgr_image = capture.retrieve(0, cv2.cv.CV_CAP_OPENNI_BGR_IMAGE)
    if not okay2:
        print "Unable to retrieve Gray Image from device"
        break

    cv2.imshow("rgb camera", bgr_image)
    unknown = depth_map.min()
    bgr_image[depth_map <= unknown] = np.zeros(3)
    bgr_image[depth_map >= depth_range] = np.zeros(3)
    cv2.imshow("rgb camera - filtered", bgr_image)
    key_pressed = cv2.waitKey(10)
    if key_pressed == 27:
        if num_frame != -1:
            with open(os.path.join(sys.argv[1], "config.txt"), 'w') as out_file:
                out_file.write("%d\n" % num_taken)
            break
        else:
            num_frame += 1
    elif key_pressed == 119:
        depth_range += 50
        print "Range increased to", depth_range
    elif key_pressed == 115:
        depth_range -= 50
        print "Range decreased to", depth_range

    if num_frame >= 0:
        num_frame += 1

    if num_frame % 3 == 0:
        num_frame = 0
        depth_map[depth_map >= depth_range] = unknown
        depth_map[depth_map > unknown] = depth_range - depth_map[depth_map > unknown] + 1
        depth_map = depth_map.reshape(-1, 1)
        print "Writing image_%d.png and depth_%d.txt to disk" % (num_taken, num_taken)
        cv2.imwrite(os.path.join(sys.argv[1], "image_%d.png" % num_taken), bgr_image)
        with open(os.path.join(sys.argv[1], "depth_%d.txt" % num_taken), 'w') as out_file:
                for depth in depth_map:
                    out_file.write("%f\n" % depth)
        num_taken += 1

cv2.destroyAllWindows()
capture.release()
