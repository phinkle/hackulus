#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv/highgui.h>

using namespace cv;

Mat applyTransformation(const Mat &p, const Mat &r, const Mat &t);
void rigid_transform_3D(const Mat &a, const Mat &b, Mat &r, Mat &t);
void nearest_neighbors(const Mat &pc_a, const Mat &pc_b, Mat &a, Mat &b);
Mat load_kinect_frame(const string image_filename, const string depth_filename);
void save_point_cloud(Mat &a, string filename);
void extractRigidTransform(const Mat& m, Mat& rotation, Mat& translation);
Mat getRigidTransform(const Mat& rotation, const Mat& translation);
Mat selectRandomPoints(const Mat& pts, double probability);
