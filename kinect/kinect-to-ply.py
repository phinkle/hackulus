import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from Queue import Queue


def icp(a, b, init_pose=(0, 0, 0), num_iters=13):
    '''
    The Iterative Closest Point estimator.
    Takes two cloudpoints a[x,y], b[x,y], an initial estimation of
    their relative pose and the number of iterations
    Returns the affine transform that transforms
    the cloudpoint a to the cloudpoint b.
    Note:
        (1) This method works for cloudpoints with minor
        transformations. Thus, the result depents greatly on
        the initial pose estimation.
        (2) A large number of iterations does not necessarily
        ensure convergence. Contrarily, most of the time it
        produces worse results.
    '''

    src = np.array([a.T], copy=True).astype(np.float32)
    dst = np.array([b.T], copy=True).astype(np.float32)

    #Initialise with the initial pose estimation
    Tr = np.array([[np.cos(init_pose[2]),-np.sin(init_pose[2]),init_pose[0]],
                   [np.sin(init_pose[2]), np.cos(init_pose[2]),init_pose[1]],
                   [0,                    0,                   1          ]])

    src = cv2.transform(src, Tr[0:2])

    for i in range(num_iters):
        #Find the nearest neighbours between the current source and the
        #destination cloudpoint
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto',
                                warn_on_equidistant=False).fit(dst[0])
        distances, indices = nbrs.kneighbors(src[0])

        #Compute the transformation between the current source
        #and destination cloudpoint
        T = cv2.estimateRigidTransform(src, dst[0, indices.T], False)
        #Transform the previous source and update the
        #current source cloudpoint
        src = cv2.transform(src, T)
        #Save the transformation from the actual source cloudpoint
        #to the destination
        Tr = np.dot(Tr, np.vstack((T,[0,0,1])))
    return Tr[0:2]


def point_cloud(depth_map, bgr_image, coords=None):
    """Create a point cloud from a disparity image and a focal length.

    Arguments:
      depth_map: depth map in mm.
      bgr_image: BGR-format left stereo image, to color the points.
      focal_length: the focal length of the stereo camera, in pixels.

    Returns:
      A string containing a PLY point cloud of the 3D locations of the
        pixels, with colors sampled from left_image. You may filter low-
        disparity pixels or noise pixels if you choose.
    """
    h, w = bgr_image.shape[:2]

    # Convert image to cloud of 3D points
    if coords is None:
        row, col = np.indices((h, w))
        row = row.reshape(-1, 1)
        col = col.reshape(-1, 1)
        indices = np.hstack([row, col])
        depth_map = -depth_map.reshape(-1, 1)
        unknown = depth_map.max()
        coords = np.hstack([indices, depth_map])
    else:
        unknown = coords[...,2].max()
    colors = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    colors = colors.reshape(-1, 3)
    pts = np.hstack([coords, colors])

    # Header for ply file
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header'''

    # Build ply file
    pts = [pt for pt in pts.tolist() if unknown - 50 < pt[2] < unknown]
    min_x, max_x = pts[0][1], pts[0][1]
    min_y, max_y = -pts[0][0], -pts[0][0]
    min_z, max_z = pts[0][2], pts[0][2]
    for pt in pts:
        min_x = min(min_x, pt[1])
        max_x = max(max_x, pt[1])
        min_y = min(min_y, -pt[0])
        max_y = max(max_y, -pt[0])
        min_z = min(min_z, pt[2])
        max_z = max(max_z, pt[2])
    print "%d <= x <= %d" % (min_x, max_x)
    print "%d <= y <= %d" % (min_y, max_y)
    print "%d <= z <= %d" % (min_z, max_z)
    ply_string = ""
    for pt in pts:
        ply_string += "%f %f %f %d %d %d\n" % (
                pt[1], -pt[0], pt[2],
                pt[3], pt[4], pt[5])

    ply_string = ply_header % dict(vert_num=len(pts)) + '\n' + ply_string

    return ply_string, np.asarray(pts)

capture = cv2.VideoCapture(cv2.cv.CV_CAP_OPENNI)
capture.set(cv2.cv.CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, cv2.cv.CV_CAP_OPENNI_VGA_30HZ)

print capture.get(cv2.cv.CV_CAP_PROP_OPENNI_REGISTRATION)
num = 0
images = []

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

    unknown = depth_map.min()
    cv2.imshow("depth map", (depth_map > unknown) * 0.5)
    cv2.imshow("rgb camera", bgr_image)
    if cv2.waitKey(10) == 27:
        pCloud, points = point_cloud(depth_map, bgr_image)
        images.append(points)
        with open("output-%03d.ply" % num, 'w') as f:
            f.write(pCloud)
        num += 1
        if num == 2:
            break

cv2.destroyAllWindows()
capture.release()
