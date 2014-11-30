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

//static int matchImage = 0;

struct KinectFrame
{
    Mat image;
    std::vector<float> depths;
    
    KinectFrame(const Mat& img, const std::vector<float>& dists)
    : image(img), depths(dists)
    {
    }
    
    float getDepth(int x, int y) const
    {
        int index = image.cols * x + y;
        return depths[index];
    }
    
    float getRed(int x, int y) const
    {
        return image.at<Vec3b>(x, y)[2];
    }
    
    float getGreen(int x, int y) const
    {
        return image.at<Vec3b>(x, y)[1];
    }
    
    float getBlue(int x, int y) const
    {
        return image.at<Vec3b>(x, y)[0];
    }
    
    Point3f getPoint(int x, int y) const
    {
        return Point3f(x, y, getDepth(x, y));
    }
};

KinectFrame getKinectFrame(string imageFilename, string depthFilename);

vector<std::pair<Point3f, Point3f> > findFeatureMatches(const KinectFrame& frameA, const KinectFrame& frameB);

Mat findTransform(const vector<std::pair<Point3f, Point3f> >& matches);

void addPlyPoints(vector<Vec<float, 6> >& ply, const KinectFrame& frame, const Mat& transform);

string posint(int x);

void print(const Mat& x)
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

int main(int argc, char** argv)
{
    string filepath = "/Users/staticsoccer/Downloads/mini-statue-2/";
    std::ifstream file(filepath + "config.txt");
    string line;
    
    getline(file, line);
    std::istringstream in(line);
    
    int files;
    in >> files;
    
    vector<KinectFrame> frames;
    
    for (int i = 0; i < files; ++i)
    {
        string index = posint(i);
//        std::cout << i << " " << index << std::endl;
        string imageFilename = filepath + "image_" + index + ".png";
        string depthFilename = filepath + "depth_" + index + ".txt";
        
        frames.push_back(getKinectFrame(imageFilename, depthFilename));
    }
    
    vector<Vec<float, 6> > ply;
    Mat lastTransform = Mat::eye(3, 4, CV_32F);
    addPlyPoints(ply, frames[0], lastTransform);
    
    for (int i = 1; i < frames.size(); ++i)
    {
        vector<std::pair<Point3f, Point3f> > matches = findFeatureMatches(frames[i - 1], frames[i]);
//        Mat transform = findTransform(matches);
//        lastTransform *= transform;
//        addPlyPoints(ply, frames[i], lastTransform);
    }
//    
//    std::ofstream plyFile;
//    
//    plyFile.open("test.ply");
//    plyFile << "ply" << std::endl;
//    plyFile << "\tformat ascii 1.0" << std::endl;
//    plyFile << "\telement vertex " << ply.size() << std::endl;
//    plyFile << "\tproperty float x" << std::endl;
//    plyFile << "\tproperty float y" << std::endl;
//    plyFile << "\tproperty float z" << std::endl;
//    plyFile << "\tproperty uchar red" << std::endl;
//    plyFile << "\tproperty uchar green" << std::endl;
//    plyFile << "\tproperty uchar blue" << std::endl;
//    plyFile << "\tend_header" << std::endl;
//    
//    for (int i = 0; i < ply.size(); ++i)
//    {
//        plyFile << ply[i][0] << ply[i][1] << ply[i][2] << round(ply[i][3]) << round(ply[i][4]) << round(ply[i][5]) << std::endl;
//    }
}

void addPlyPoints(vector<Vec<float, 6> >& ply, const KinectFrame& frame, const Mat& transform)
{
    vector<Point3f> inPts, outPts;
    for (int i = 0; i < frame.image.rows; ++i)
    {
        for (int j = 0; j < frame.image.cols; ++j)
        {
            inPts.push_back(frame.getPoint(i, j));
        }
    }
    
    perspectiveTransform(inPts, outPts, transform);
    
    int index = 0;
    for (int i = 0; i < frame.image.rows; ++i)
    {
        for (int j = 0; j < frame.image.cols; ++j, ++index)
        {
            int red = frame.getRed(i, j);
            int blue = frame.getBlue(i, j);
            int green = frame.getGreen(i, j);

            Vec<float, 6> point;
            point[0] = outPts[index].x;
            point[1] = outPts[index].y;
            point[2] = outPts[index].z;
            point[3] = red;
            point[4] = green;
            point[5] = blue;
            
            ply.push_back(point);
        }
    }
}

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

Mat findTransform(const vector<std::pair<Point3f, Point3f> >& matches)
{
    Point3f centroidA(0.0f, 0.0f, 0.0f), centroidB(0.0f, 0.0f, 0.0f);
    
    for (int i = 0; i < matches.size(); ++i)
    {
        centroidA += matches[i].first;
        centroidB += matches[i].second;
    }
    
    centroidA *= 1.0f / matches.size();
    centroidB *= 1.0f / matches.size();
    
    Mat h(3, 3, CV_32F);
    print(h);
    
    for (int i = 0; i < matches.size(); ++i)
    {
        Point3f transA = matches[i].first - centroidA;
        Point3f transB = matches[i].second - centroidB;
        
        for (int j = 0; j < 3; ++j)
        {
            for (int k = 0; k < 3; ++k)
            {
                h.at<float>(i, j) += get(transA, j) * get(transB, k);
            }
        }
    }
    
    Mat w, u, vt;
    SVD::compute(h, w, u, vt);
    
    float centroidAData[3][1] =
    {
        {centroidA.x},
        {centroidA.y},
        {centroidA.z}
    };
    
    float centroidBData[3][1] =
    {
        {centroidA.x},
        {centroidB.y},
        {centroidB.z}
    };
    
    Mat centroidAMatrix(3, 1, CV_32F, centroidAData);
    Mat centroidBMatrix(3, 1, CV_32F, centroidBData);
    
    Mat rotation = h.t() * vt.t();
    Mat translation = -rotation * centroidAMatrix + centroidBMatrix;
    translation = translation.t();
    
    Mat transform(4, 3, CV_32F);
    
    rotation.row(0).copyTo(transform.row(0));
    rotation.row(1).copyTo(transform.row(1));
    rotation.row(2).copyTo(transform.row(2));
    translation.row(0).copyTo(transform.row(3));
    
    return transform.t();
}

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

float dot(const Point2f& a, const Point2f b)
{
    return a.x * b.x + a.y * b.y;
}

float length(const Point2f& a)
{
    return sqrt(a.x * a.x + a.y * a.y);
}

float cosine(const Point2f& a, const Point2f b)
{
    float result = dot(a, b) / (length(a) * length(b));
    return result;
}

vector<std::pair<Point3f, Point3f> > findFeatureMatches(const KinectFrame& frameA, const KinectFrame& frameB)
{
    string filepath = "/Users/staticsoccer/Downloads/mini-statue-2/";

    Mat grayA;
    Mat grayB;
    
    cvtColor(frameA.image, grayA, CV_BGR2GRAY);
    cvtColor(frameB.image, grayB, CV_BGR2GRAY);
    
    int minHessian = 400;
    
    SurfFeatureDetector detector(minHessian);
    std::vector<KeyPoint> kpA, kpB;
    
    detector.detect(grayA, kpA);
    detector.detect(grayB, kpB);
    
    //    Mat kpImageA; Mat kpImageB;
    
    //    drawKeypoints(imageA, kpA, kpImageA);
    //    drawKeypoints(imageB, kpB, kpImageB);
    
    //    imshow("Keypoints A", kpImageA);
    //    imshow("Keypoints B", kpImageB);
    
    SurfDescriptorExtractor extractor;
    Mat descA, descB;
    
    extractor.compute(grayA, kpA, descA);
    extractor.compute(grayB, kpB, descB);
    
    BFMatcher matcher;
    std::vector<DMatch> matches;
    
    matcher.match(descA, descB, matches);
    
    double minDist = matches[0].distance;
    double maxDist = matches[0].distance;
    
    for (int i = 1; i < matches.size(); ++i)
    {
        double dist = matches[i].distance;
        
        if (dist > maxDist)
        {
            maxDist = dist;
        }
        
        if (dist < minDist)
        {
            minDist = dist;
        }
    }
    
    std::vector<DMatch> goodMatches;
    
    Point2f left(-1.0f, 0.0f);
    Point2f right(1.0f, 0.0f);
    float threshold = 0.975f;
    
    for (int i = 0; i < matches.size(); ++i)
    {
        const Point2f& pA = kpA[matches[i].queryIdx].pt;
        const Point2f& pB = kpB[matches[i].trainIdx].pt;
        
        const Point2f p = pA - pB;
        
//        std::cout << p.x << " " << p.y << ": " << cosine(p, left) << ", " <<  cosine(p, right) << std::endl;
        
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
            if (matches[i].distance)
            goodMatches.push_back(matches[i]);
        }
    }
    
    vector<std::pair<Point3f, Point3f> > pointMatches;
    
    for (int i = 0; i < goodMatches.size(); ++i)
    {
        const Point2f& pA = kpA[goodMatches[i].queryIdx].pt;
        const Point2f& pB = kpB[goodMatches[i].trainIdx].pt;
        
        const Point3f p3A = frameA.getPoint(round(pA.x), round(pA.y));
        const Point3f p3B = frameB.getPoint(round(pB.x), round(pB.y));
        
        pointMatches.push_back(std::pair<Point3f, Point3f>(p3A, p3B));
    }
    
//    Mat imageMatches;
//    drawMatches(frameA.image, kpA, frameB.image, kpB, goodMatches, imageMatches, Scalar::all(-1), Scalar::all(-1),
//                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
//    imshow("Matches", imageMatches);
//    imwrite(filepath + "imageMatch_" + posint(matchImage++) + ".png", imageMatches);
//    waitKey(0);
    
    return pointMatches;
}

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

/**
 * Reads in a .ply file into a std::Vector, where its elements
 * are cv:Vec's of 6 floats.
 *
 * @param filename the filename of the .ply file.
 * @param headerLength the number of lines representing the header, or in other words,
 * the number of lines to ignore.
 * @param points3D the vector to receive the point cloud
 *
 * @return true if the points were successfully retrieved; otherwise false.
 */
bool readInPly(string filename, int headerLength, vector<Vec<float, 6> >& points3D)
{
    std::ifstream file(filename);
    string line;
    
    if (!file.is_open())
    {
        return false;
    }
    
    // Header description of input
    for (int i = 0; i < 10; ++i)
    {
        getline(file, line);
    }
    
    // Retreival of all vertices into a std::vector
    while (getline(file, line))
    {
        std::istringstream in(line);
        
        float x, y, z, r, g, b;
        in >> x >> y >> z >> r >> g >> b;
        
        Vec<float, 6> point;
        point[0] = x;
        point[1] = y;
        point[2] = z;
        point[3] = r;
        point[4] = g;
        point[5] = b;
        
        points3D.push_back(point);
    }
    
    file.close();
    return true;
}
