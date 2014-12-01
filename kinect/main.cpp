//
//  main.cpp
//  opencv
//
//  Created by Jaime Rivera on 11/29/14.
//  Copyright (c) 2014 Jaime Rivera. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <utility>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv/highgui.h>

using namespace cv;

static float epsilon = 0.000001f;
static float errorThreshold = 5.0f;
static float pixelDistance = 4.0f;
static float pi = 3.141592653f;
static int matchImage = 0;
static string filepath = "/Users/staticsoccer/Downloads/statue-fixed-inverted/";

struct KinectFrame
{
    Mat image;
    std::vector<float> depths;
    
    KinectFrame(Mat img, const std::vector<float>& dists)
    : image(img), depths(dists)
    {
    }
    
    int getDepth(int x, int y) const
    {
        int index = image.cols * x + y;
        return depths[index];
    }
    
    int getRed(int x, int y) const
    {
        return image.at<Vec3b>(x, y)[2];
    }
    
    int getGreen(int x, int y) const
    {
        return image.at<Vec3b>(x, y)[1];
    }
    
    int getBlue(int x, int y) const
    {
        return image.at<Vec3b>(x, y)[0];
    }
    
    Point3i getPoint(int x, int y) const
    {
        return Point3i(x, y, getDepth(x, y));
    }
};

// No-formatting Matrix print
void print(Mat x)
{
    for (int i = 0; i < x.rows; ++i)
    {
        for (int j = 0; j < x.cols; ++j)
        {
            std::cout << x.at<float>(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

// Lazy instance var retrieval
float get(const Point3f& p, int i)
{
    if (i == 0)
    {
        return p.x;
    }
    else if (i == 1)
    {
        return p.y;
    }
    else if (i == 2)
    {
        return p.z;
    }
    else
    {
        throw Exception();
    }
}

// Turns a positive integer into a string
string posint(int x)
{
    if (x == 0)
    {
        return "0";
    }
    
    string s = "";
    
    while (x > 0)
    {
        s = (char)('0' + (x % 10)) + s;
        x /= 10;
    }
    
    return s;
}

// Point dot product
float dot(const Point2f& a, const Point2f b)
{
    return a.x * b.x + a.y * b.y;
}

// Point magnitude
float length(const Point2f& a)
{
    return sqrt(a.x * a.x + a.y * a.y);
}

// Point cosine-similarity
float cosine(const Point2f& a, const Point2f b)
{
    float result = dot(a, b) / (length(a) * length(b));
    return result;
}

// Transforms points based on rotation and translation
void transformPoints(const Mat& input, const Mat& rotation, const Mat& translation, Mat& output)
{
    output = input * rotation;
    
    for (int i = 0; i < input.rows; ++i)
    {
        output.row(i) += translation;
    }
}

void findCentroid(const Mat& m, Mat& centroid)
{
    centroid = Mat::zeros(1, 3, CV_32F);
    
    for (int i = 0; i < m.rows; ++i)
    {
        centroid += m.row(i);
    }
    
    centroid *= 1.0f / m.rows;
}

// Adds a frame's image based on the rotation and translation to a .ply file.
void addPlyPoints(vector<Vec<float, 6> >& ply, const KinectFrame& frame, Mat rotation, Mat translation)
{
    
    float data[1][3];
    
    for (int i = 0; i < frame.image.rows; ++i)
    {
        for (int j = 0; j < frame.image.cols; ++j)
        {
            int red = frame.getRed(i, j);
            int blue = frame.getBlue(i, j);
            int green = frame.getGreen(i, j);
            
            const Point3i& p = frame.getPoint(i, j);
            
            if (p.z <= epsilon)
            {
                continue;
            }
            
            data[0][0] = p.x;
            data[0][1] = p.y;
            data[0][2] = p.z;
            
            Mat pMat(1, 3, CV_32F, data);
            Mat transformedPoint = pMat * rotation + translation;
            
            Vec<float, 6> point;
            point[0] = float(transformedPoint.at<float>(0, 0));
            point[1] = float(transformedPoint.at<float>(0, 1));
            point[2] = float(transformedPoint.at<float>(0, 2));
            point[3] = red;
            point[4] = green;
            point[5] = blue;
            
            ply.push_back(point);
        }
    }
}

// Finds the rigid transform, rotation and translation, that moves matchA to matchB.
void findTransform(const Mat& matchA, const Mat& matchB, Mat& rot, Mat& trans, Mat& centroidA, Mat& centroidB)
{
    assert(matchA.size == matchB.size);
    
    Mat mA = matchA.clone();
    Mat mB = matchB.clone();
    
    findCentroid(mA, centroidA);
    findCentroid(mB, centroidB);
    
    for (int i = 0; i < mA.rows; ++i)
    {
        mA.row(i) -= centroidA;
        mB.row(i) -= centroidB;
    }

    Mat h = mA.t() * mB;
    
    Mat w, u, vt;
    SVD::compute(h, w, u, vt);
    
    Mat rotation = vt.t() * u.t();
    
    if (determinant(rotation) < 0.0f)
    {
        rotation.col(2) = -rotation.col(2);
    }
    std::cout << "Determinant:\t" << determinant(rotation) << std::endl;
    
    Mat translation = -rotation * centroidA.t() + centroidB.t();
    
    rot = rotation.t();
    trans = translation.t();
}

// Finds corresponding 3D points using the kinect's rgb images and depth parameters.
void findFeatureMatches(const KinectFrame& frameA, const KinectFrame& frameB, Mat& outA, Mat& outB)
{
    Mat grayA;
    Mat grayB;
    
    cvtColor(frameA.image, grayA, CV_BGR2GRAY);
    cvtColor(frameB.image, grayB, CV_BGR2GRAY);
    
    
    Ptr<FeatureDetector> detector;
    detector = new DynamicAdaptedFeatureDetector (new FastAdjuster(10, true), 25, 50, 5);
    std::vector<KeyPoint> kpA, kpB;
    
    detector->detect(grayA, kpA);
    detector->detect(grayB, kpB);
    
    
    Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");
    Mat descA, descB;
    
    extractor->compute(grayA, kpA, descA);
    extractor->compute(grayB, kpB, descB);
    
//    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
//    vector<vector<DMatch> > matches;
//    
//    matcher->knnMatch(descA, descB, matches, 1);
    
    BFMatcher matcher(NORM_L2);
    std::vector<DMatch> matches;
    matcher.match(descA, descB, matches);
    
    std::vector<DMatch> goodMatches;
    
    Point2f left(-1.0f, 0.0f);
    Point2f right(1.0f, 0.0f);
    float threshold = 0.999f;
    
    for (int i = 0; i < matches.size(); ++i)
    {
        Point2f from = kpA[matches[i].queryIdx].pt;
        Point2f to = kpB[matches[i].trainIdx].pt;
        
        const Point2f p = from - to;
        
        float l = cosine(p, left);
        float r = cosine(p, right);
        
        if (l < 0.0f)
        {
            l = -l;
        }
        
        if (r < 0.0f)
        {
            r = -r;
        }

        if (l >= threshold || r >= threshold)
        {
            goodMatches.push_back(matches[i]);
        }
    }
    
    // The maximum number of pixels to be apart
    
//    for (size_t i = 0; i < matches.size(); ++i)
//    {
//        for (int j = 0; j < matches[i].size(); ++j)
//        {
//            Point2f from = kpA[matches[i][j].queryIdx].pt;
//            Point2f to = kpB[matches[i][j].trainIdx].pt;
//            
//            const Point2f p = from - to;
//            
//            float l = cosine(p, left);
//            float r = cosine(p, right);
//            
//            if (l < 0.0f)
//            {
//                l = -l;
//            }
//            
//            if (r < 0.0f)
//            {
//                r = -r;
//            }
//
//            //calculate local distance for each possible match
//            double dist = length(p);
//            
//            //save as best match if local distance is in specified area and on same height
//            if (dist < pixelDistance && (l >= threshold || r >= threshold))
//            {
//                goodMatches.push_back(matches[i][j]);
//                j = (int) matches[i].size();
//            }
//        }
//    }
    
    outA = Mat::zeros((int) goodMatches.size(), 3, CV_32F);
    outB = Mat::zeros((int) goodMatches.size(), 3, CV_32F);
    
    for (int i = 0; i < goodMatches.size(); ++i)
    {
        const Point2f& pA = kpA[goodMatches[i].queryIdx].pt;
        const Point2f& pB = kpB[goodMatches[i].trainIdx].pt;
        
        const Point3f p3A = frameA.getPoint(round(pA.x), round(pA.y));
        const Point3f p3B = frameB.getPoint(round(pB.x), round(pB.y));
        
        for (int j = 0; j < 3; ++j)
        {
            outA.at<float>(i, j) = get(p3A, j);
            outB.at<float>(i, j) = get(p3B, j);
        }
    }
    
    Mat imageMatches;
    drawMatches(frameA.image, kpA, frameB.image, kpB, goodMatches, imageMatches, Scalar::all(-1), Scalar::all(-1),
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imshow("Matches", imageMatches);
    imwrite(filepath + "imageMatch_" + posint(matchImage++) + ".png", imageMatches);
    waitKey(0);
}

// Stores an rgb image with depth information as a Xbox Kinect frame.
KinectFrame getKinectFrame(string imageFilename, string depthFilename)
{
    Mat image;
    image = imread(imageFilename);
    
    if (!image.data)
    {
        std::cout << "Could not read " << imageFilename << std::endl;
        throw Exception();
    }
    
    vector<float> depths;
    
    std::ifstream file(depthFilename);
    
    float depth;
    string line;
    while (getline(file, line))
    {
        std::istringstream in(line);
        in >> depth;
        depths.push_back(depth);
    }
    
    return KinectFrame(image, depths);
}

double error(const Mat& a, const Mat& b)
{
    assert(a.size() == b.size());
    Mat e = a - b;
    multiply(e, e, e);
    return sum(e)[0] / (double) a.rows ;
}

void applyXZRotation(float radians, Mat& out)
{
    out = Mat::zeros(3, 3, CV_32F);
    
    out.at<float>(0, 0) = cos(radians);
    out.at<float>(2, 2) = cos(radians);
    out.at<float>(2, 0) = -sin(radians);
    out.at<float>(0, 2) = sin(radians);
    out.at<float>(1, 1) = 1.0f;
}

float angleToRadians(float angle)
{
    return angle * pi / 180.0f;
}

float absoluteDiff(float a, float b)
{
    float c = a - b;
    
    return (c < 0.0f) ? -c : c;
}

void findXZRotation(const Mat& matchA, const Mat& matchB, Mat& out)
{
    assert(matchA.size == matchB.size);
    
    Mat mA = matchA.clone();
    Mat mB = matchB.clone();
    Mat centroidA, centroidB;
    
    findCentroid(matchA, centroidA);
    findCentroid(matchB, centroidB);
    
    for (int i = 0; i < mA.rows; ++i)
    {
        mA.row(i) -= centroidA;
        mB.row(i) -= centroidB;
    }
    
    float angleOffset = 30;
    
    float lo = -angleOffset;
    float hi = angleOffset;
    
    while (absoluteDiff(lo, hi) > epsilon * 100)
    {
        float step = (hi - lo) / 3.0f;
        
        float x = lo + step;
        float y = lo + step + step;
        
        Mat rot;
        
        applyXZRotation(angleToRadians(x), rot);
        float xError = error(mA * rot, mB);
        
        applyXZRotation(angleToRadians(y), rot);
        float yError = error(mA * rot, mB);
        
        if (xError > yError)
        {
            lo = x;
        }
        else
        {
            hi = y;
        }
    }
    
    applyXZRotation(angleToRadians(lo), out);
}

// Tests the validity of the SVD computation.
void test()
{
//    cv::theRNG().state = getTickCount();

    int n = 1000;
    Mat A = Mat::zeros(n, 3, CV_32F);
    Mat R = Mat::zeros(3, 3, CV_32F);
    Mat T = Mat::zeros(1, 3, CV_32F);
    
    randn(A, 0, 1);
//    R.at<float>(0, 0) = cos(.5);
//    R.at<float>(2, 2) = cos(.5);
//    R.at<float>(2, 0) = -sin(.5);
//    R.at<float>(0, 2) = sin(.5);
//    R.at<float>(1, 1) = 1.0f;
//    randn(R, 0, 1);
    randn(T, 0, 1);
    
    Mat B = A * R;
    
    for (int i = 0; i < n; ++i)
    {
        B.row(i) += T;
    }

    Mat rot, trans, cA, cB;
    findTransform(A, B, rot, trans, cA, cB);
    
    Mat C = A.clone();
    
    for (int i = 0; i < n; ++i)
    {
//        C.row(i) -= cB;
    }
    
    C = C * rot;
    
    for (int i = 0; i < n; ++i)
    {
        C.row(i) += trans;
    }
    
    print(rot);
    std::cout << std::endl;
    print(R);
    std::cout << std::endl;
    print(T);
    std::cout << std::endl;
    print(trans);
    std::cout << std::endl;
    
    std::cout << "Total: " << error(C, B) << "; Rotation: " << error(rot, R) << "; Translation: " << error(trans, T) << std::endl;
    std::cout << std::endl;
}

int main(int argc, char** argv)
{
    std::ifstream file(filepath + "config.txt");
    string line;
    
    getline(file, line);
    std::istringstream in(line);
    
    int files;
    in >> files;
    
    vector<KinectFrame> frames;
    
    int skipCount = 0;
    for (int i = 0; i < files; ++i)
    {
        string index = posint(i);
        string imageFilename = filepath + "image_" + index + ".png";
        string depthFilename = filepath + "depth_" + index + ".txt";
        
        frames.push_back(getKinectFrame(imageFilename, depthFilename));
    }
    
    // This will be the entire generated .ply
    vector<Vec<float, 6> > ply;
    
    // Rotation is defaulted to the Identity 3x3
    Mat rotation = Mat::eye(3, 3, CV_32F);
    
    // There has been no translation, yet, so (0.0, 0.0, 0.0)
    Mat translation = Mat::zeros(1, 3, CV_32F);
    
    // Add the first frame to be the initial set of points in the .ply
    addPlyPoints(ply, frames[0], rotation, translation);
    
    for (int i = 1; i < frames.size(); ++i)
    {
        // Find sets of correspondences between the RGB images
        // of an adjacent pair of kinect frames.
        Mat mA, mB;
        findFeatureMatches(frames[i - 1], frames[i], mA, mB);
        
        // If we had no matches, let's not include this set to reduce
        // noise in our model.
        if (mA.rows == 0)
        {
            ++skipCount;
            continue;
        }
        
        // We determined how to move the previous set of points with this
        // given rotation and translation, so move the points once again
        // so that this new set of correspondences can move find the
        // proper rigid transform.
        Mat matchedMoved;
        transformPoints(mA, rotation, translation, matchedMoved);
        
        // Find the rigid transform from the new batch of points, to the
        // previously moved set of points. This should rotate and translate
        // the image to align with the previous set. So, given a continuous
        // rotation, the point cloud should rotate properly.
        Mat rot, trans, cA, cB;
        findTransform(mB, matchedMoved, rot, trans, cA, cB);
//        findXZRotation(mB, matchedMoved, rot);
//        trans = Mat::zeros(3, 1, CV_32F);
        
//        std::cout << std::endl;
//        print(rot);
//        std::cout << std::endl;
//        print(trans);
//        std::cout << std::endl;
        
        Mat newMoved;
        transformPoints(mB, rot, trans, newMoved);
        
        double err = error(newMoved, matchedMoved);
        std::cout << "Error[" << i << "]:\t" << err << "\t(" << mB.rows << " samples)";
        
        
        if (err > errorThreshold)
        {
            std::cout << "\t!skipped" << std::endl;
            ++skipCount;
            continue;
        }
        std::cout << std::endl;
        
        // TODO: We're not moving the kinect, so it's not a simple transformation from one
        // to the next. We need to either keep track of the previous positions that we
        // just added, or some other means..
        addPlyPoints(ply, frames[i], rot, trans);
        
        // Keep track of this rotation and translation, so that we can move the
        // next used frame by its previous rigid transform.
        rotation = rot;
        translation = trans;
    }
    
    std::cout << "Skipped:\t" << skipCount << " / " << frames.size() << std::endl;
    
    std::fstream plyFile;
    
    plyFile.open(filepath + "test.ply", std::fstream::out);
    plyFile << "ply" << std::endl;
    plyFile << "format ascii 1.0" << std::endl;
    plyFile << "element vertex " << ply.size() << std::endl;
    plyFile << "property float x" << std::endl;
    plyFile << "property float y" << std::endl;
    plyFile << "property float z" << std::endl;
    plyFile << "property uchar red" << std::endl;
    plyFile << "property uchar green" << std::endl;
    plyFile << "property uchar blue" << std::endl;
    plyFile << "end_header" << std::endl;
    
    for (int i = 0; i < ply.size(); ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            plyFile << ply[i][j] << " ";
        }
        
        for (int j = 0; j < 2; ++j)
        {
            plyFile << round(ply[i][j + 3]) << " ";
        }
        
        plyFile << round(ply[i][5]) << std::endl;
    }
    
    plyFile.close();
}
