import sys
import cv2
import numpy as np


def print_usage():
    print "Usage: %s src_file dst_file out_file" % sys.argv[0]


def ply_to_img(ply):
    """Assuming ply is format ascii 1.0 and vertices are formatted like so:
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    
    Arguments:
        ply: point cloud in ply format as a string
    """
    ply = ply.splitlines()[10:]
    for line in ply:
        pass
    pass


def main():
    if len(sys.argv < 4):
        printUsage()
    src_file = sys.argv[1]
    dst_file = sys.argv[2]
    out_file = sys.argv[3]



if __name__ == '__main__':
    main()
