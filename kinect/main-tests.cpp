#include "icp.hpp"

#define EPSILON 0.001f

using namespace cv;

float errorDifference(const Mat& a, const Mat& b)
{
	Mat difference = a - b;
	Mat sqDifference = difference * difference;

	return sum(sqDifference)[0];
}

void testApplyTransformation()
{
	Mat point = (Mat_<float>(3, 1) << 1, 0, 0);
	Mat rotation = (Mat_<float>(3, 3) << cos(90), 0, -sin(90), 0, 0, 0, sin(90), 0, cos(90));
	Mat translation = (Mat_<float>(3, 1) << 1, 1, 1);

	Mat transformedPoint = applyTransformation(point, rotation, translation);
	Mat correctPoint = (Mat_<float>(3, 1) << 2, 1, 1);

	assert (errorDifference(point, correctPoint) <= EPSILON);
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

	assert(errorDifference(rotation, r) <= EPSILON);
	assert(errorDifference(translation, t) <= EPSILON);
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
		for (int j = 0; j < 3; ++j)
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
		assert(m.at<float>(3, i) == t.at<float>(i, 0));
		for (int j = 0; j < 3; ++j)
		{
			assert(m.at<float>(i, j) == r.at<float>(i, j));
		}
	}
}

int main()
{
	testApplyTransformation();
	testNearestNeighbors();
	testRigidTransform3d();
	testGetRigidTransform();
	testExtractRigidTransform();
  return 0;
}
