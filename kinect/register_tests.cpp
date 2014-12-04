#include "register.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv/highgui.h>

using namespace cv;

float errorDifference(const Mat& a, const Mat& b)
{
	Mat difference = point - correctPoint;
	Mat sqDifference = difference * difference;

	return sum(sqDifference);
}

void testApplyTransformation()
{
	// Mat applyTransformation(const Mat &p, const Mat &r, const Mat &t);

	float epsilon = 0.001f;

	Mat point = (Mat_<float>(3, 1) << 1, 0, 0);
	Mat rotation = (Mat_<float>(3, 3) << cos(90), 0, -sin(90), 0, 0, 0, sin(90), 0, cos(90));
	Mat translation = (Mat_<float>(3, 1) << 1, 1, 1);

	Mat transformedPoint = applyTransformation(point, rotation, translation);
	Mat correctPoint = (Mat_float<3, 1> << 2, 1, 1);

	assert(errorDifference(point, correctPoint) <= epsilon);
}

void testNearestNeighbors()
{
	// void nearest_neighbors(const Mat &pc_a, const Mat &pc_b, Mat &a, Mat &b);

	Mat setA = (Mat_<float>(3, 3) << 1, 1, 1, 4, 4, 4, 7, 7, 7);
	Mat setB = (Mat_<float>(3, 3) << 0, 0, 0, 3, 3, 3, 6, 6, 6);

	Mat a, b;
	nearest_neighbors(setA, setB, a, b);

	assert(countNonZero(setA != a) == 0);
	assert(countNonZero(setB != b) == 0);
}

void testRigidTransform3d()
{
	// void rigid_transform_3D(const Mat &a, const Mat &b, Mat &r, Mat &t);

	Mat a = Mat::eye(4, 4, CV_32F);
	Mat rotation = (Mat_<float>(3, 3) << cos(90), 0, -sin(90), 0, 0, 0, sin(90), 0, cos(90));
	Mat translation = (Mat_<float>(3, 1) << 1, 1, 1);

	Mat b = applyTransformation(a, rotation, translation);

	Mat r, t;
	rigid_transform_3D(a, b, r, t);

	float epsilon = 0.001f;

	assert(errorDifference(rotation, r) <= epsilon);
	assert(errorDifference(translation, t) <= epsilon);
}

void testGetRigidTransform()
{
	// Mat getRigidTransform(const Mat& rotation, const Mat& translation);

	Mat r1 = Mat(3, 3, CV_32F);
	Mat t1 = Mat(3, 1, CV_32F);

	for (int i = 0; i < 3; ++i)
	{
		t1.at<float>(i, 0) = i;
		for (int j = 0; j < 3; ++j)
		{
			r1.at<float>(i, j) = i;
		}
	}

	Mat rt1 = getRigidTransform(r1, t1);

	for (int i = 0; i < 3; ++i)
	{
		assert(rt1.at<float>(3, i) == t1.at<float>(i, 0));
		for (int j = 0; j < 3; +j)
		{
			assert(rt1.at<float>(i, j) == r1.at<float>(i, j));
		}
	}
}

void testExtractRigidTransform()
{
	// void extractRigidTransform(const Mat& m, Mat& rotation, Mat& translation);
	Mat m = Mat::eye(4, 4, CV_32F);

	Mat r, t;
	extractRigidTransform(m, r, t);

	for (int i = 0; i < 3; ++i)
	{
		assert(rt1.at<float>(3, i) == t1.at<float>(i, 0));
		for (int j = 0; j < 3; +j)
		{
			assert(rt1.at<float>(i, j) == r1.at<float>(i, j));
		}
	}
}
