import cv2
import os, sys
import numpy as np
from Queue import Queue


capture = cv2.VideoCapture(cv2.cv.CV_CAP_OPENNI)
capture.set(cv2.cv.CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, cv2.cv.CV_CAP_OPENNI_VGA_30HZ)

print capture.get(cv2.cv.CV_CAP_PROP_OPENNI_REGISTRATION)
num_frame = -1
num_taken = 0

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

    depth_map = -depth_map
    unknown = depth_map.max()
    bgr_image[depth_map < unknown - 50] = np.zeros(3)
    bgr_image[depth_map >= unknown] = np.zeros(3)
    cv2.imshow("rgb camera", bgr_image)
    if cv2.waitKey(10) == 27:
        if num_frame != -1:
            break
        else:
            num_frame += 1

    if num_frame >= 0:
        num_frame += 1

    if num_frame % 10 == 0:
        num_frame = 0
        depth_map[depth_map < unknown - 50] = unknown
        depth_map = depth_map.reshape(-1, 1)
        print "Writing image_%d.png and depth_%d.txt to disk" % (num_taken, num_taken)
        cv2.imwrite(os.path.join(sys.argv[1], "image_%d.png" % num_taken), bgr_image)
        with open(os.path.join(sys.argv[1], "depth_%d.txt" % num_taken), 'w') as out_file:
                for depth in depth_map:
                    out_file.write("%f\n" % depth)
        num_taken += 1

cv2.destroyAllWindows()
capture.release()
