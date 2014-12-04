#include "icp.hpp"

#define EPSILON 0.001f

float errorDifference(const Mat& a, const Mat& b)
{
	Mat difference = a - b;
	return difference.dot(difference);
}

void testApplyTransformation()
{
	float deg90 = 1.57079633f;

	Mat point = (Mat_<float>(1, 6) << 1, 0, 0, 0, 0, 0);
	Mat rotation = (Mat_<float>(3, 3) << cos(deg90), 0, -sin(deg90), 0, 0, 0, sin(deg90), 0, cos(deg90));
	Mat translation = (Mat_<float>(3, 1) << 1, 1, 1);

	Mat transformedPoint = applyTransformation(point, rotation, translation);
	Mat correctPoint = (Mat_<float>(1, 6) << 1, 1, 2, 0, 0, 0);

	assert (errorDifference(transformedPoint, correctPoint) <= EPSILON);
}

void testNearestNeighbors()
{
	Mat setA = (Mat_<float>(3, 6) << 1, 1, 1, 0, 0, 0, 100, 100, 100, 0, 0, 0, 7, 7, 7, 0, 0, 0);
	Mat setB = (Mat_<float>(3, 6) << 0, 0, 0, 0, 0, 0, 99, 99, 99, 0, 0, 0, 6, 6, 6, 0, 0, 0);

	Mat a, b;
	nearest_neighbors(setA, setB, a, b);

	assert(errorDifference(setB.colRange(0, 3), b) <= EPSILON);
}

void testRigidTransform3d()
{
	// void rigid_transform_3D(const Mat &a, const Mat &b, Mat &r, Mat &t);
	float deg90 = 1.57079633f;
	Mat a = Mat::eye(3, 3, CV_32F);

	Mat rotation = (Mat_<float>(3, 3) << cos(deg90), 0, -sin(deg90), 0, 0, 0, sin(deg90), 0, cos(deg90));
	Mat translation = (Mat_<float>(3, 1) << 1, 1, 1);

	Mat b = applyTransformation(a, rotation, translation);

	Mat r, t;
	rigid_transform_3D(a, b, r, t);

	Mat c = applyTransformation(a, r, t);

	// assert(errorDifference(a, b) <= EPSILON);
}

void testGetRigidTransform()
{
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
		assert(rt1.at<float>(i, 3) == t1.at<float>(i, 0));

		for (int j = 0; j < 3; ++j)
		{
			assert(rt1.at<float>(i, j) == r1.at<float>(i, j));
		}
	}
}

void testExtractRigidTransform()
{
	Mat m = Mat::eye(4, 4, CV_32F);

	Mat r, t;
	extractRigidTransform(m, r, t);

	for (int i = 0; i < 3; ++i)
	{
		assert(m.at<float>(i, 3) == t.at<float>(i, 0));
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
