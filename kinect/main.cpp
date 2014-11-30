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
static int matchImage = 0;
static string filepath = "/Users/staticsoccer/Downloads/statue-fixed/";

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

KinectFrame getKinectFrame(string imageFilename, string depthFilename);

vector<std::pair<Point3f, Point3f> > findFeatureMatches(const KinectFrame& frameA, const KinectFrame& frameB);

Mat findTransform(const vector<std::pair<Point3f, Point3f> >& matches);

void addPlyPoints(vector<Vec<float, 6> >& ply, const KinectFrame& frame, Mat transform);

string posint(int x);

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

int main(int argc, char** argv)
{
    std::ifstream file(filepath + "config.txt");
    string line;
    
    getline(file, line);
    std::istringstream in(line);
    
    int files;
    in >> files;
    
    vector<KinectFrame> frames;
    
    for (int i = 0; i < files ; ++i)
    {
        string index = posint(i);
        string imageFilename = filepath + "image_" + index + ".png";
        string depthFilename = filepath + "depth_" + index + ".txt";
        
        frames.push_back(getKinectFrame(imageFilename, depthFilename));
    }
    
    vector<Vec<float, 6> > ply;
    Mat lastTransform = Mat::eye(4, 4, CV_32F);
    addPlyPoints(ply, frames[0], lastTransform);
    
    for (int i = 1; i < frames.size(); ++i)
    {
        vector<std::pair<Point3f, Point3f> > matches = findFeatureMatches(frames[i - 1], frames[i]);
        
        if (matches.empty())
        {
            continue;
        }
        
        Mat transform = findTransform(matches);
        lastTransform = transform * lastTransform;
        addPlyPoints(ply, frames[i], lastTransform);
    }
    
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

void addPlyPoints(vector<Vec<float, 6> >& ply, const KinectFrame& frame, Mat transform)
{
    
    float data[4][1];
    data[3][0] = 1.0f;
    
    int index = 0;
    for (int i = 0; i < frame.image.rows; ++i)
    {
        for (int j = 0; j < frame.image.cols; ++j, ++index)
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
            data[1][0] = p.y;
            data[2][0] = p.z;
            
            Mat pMat(4, 1, CV_32F, data);
            
            Mat transformedPoint = transform * pMat;

            Vec<float, 6> point;
            point[0] = transformedPoint.at<float>(0, 0);
            point[1] = transformedPoint.at<float>(1, 0);
            point[2] = transformedPoint.at<float>(2, 0);
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
    
    Mat h = Mat::zeros(3, 3, CV_32F);
    
    for (int i = 0; i < matches.size(); ++i)
    {
        Point3f transA = matches[i].first - centroidA;
        Point3f transB = matches[i].second - centroidB;
        
        for (int j = 0; j < 3; ++j)
        {
            for (int k = 0; k < 3; ++k)
            {
                h.at<float>(j, k) += get(transA, j) * get(transB, k);
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
    
    Mat rotation = vt.t() * u.t();
    Mat translation = -rotation * centroidAMatrix + centroidBMatrix;
    
    Mat transform(4, 4, CV_32F);
    
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            transform.at<float>(i, j) = float(rotation.at<float>(i, j));
        }
    }
    
    for (int i = 0; i < 3; ++i)
    {
        transform.at<float>(i, 3) = float(translation.at<float>(i, 0));
        transform.at<float>(3, i) = 0.0f;
    }
    
    transform.at<float>(3, 3) = 1.0f;
    
    return transform;
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
    Mat grayA;
    Mat grayB;
    
    cvtColor(frameA.image, grayA, CV_BGR2GRAY);
    cvtColor(frameB.image, grayB, CV_BGR2GRAY);
    
    
    Ptr<FeatureDetector> detector;
    detector = new DynamicAdaptedFeatureDetector ( new FastAdjuster(10,true), 5000, 10000, 10);
    std::vector<KeyPoint> kpA, kpB;
    
    detector->detect(grayA, kpA);
    detector->detect(grayB, kpB);

    
    Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");
    Mat descA, descB;
    
    extractor->compute(grayA, kpA, descA);
    extractor->compute(grayB, kpB, descB);
    
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
    vector<vector<DMatch> > matches;
    
    matcher->knnMatch(descA, descB, matches, 500);
    
    std::vector<DMatch> goodMatches;
    
    Point2f left(-1.0f, 0.0f);
    Point2f right(1.0f, 0.0f);
    float threshold = 0.99f;
    
    double tresholdDist = 0.05 * sqrt(double(grayA.size().height*grayA.size().height + grayA.size().width*grayA.size().width));
    
    for (size_t i = 0; i < matches.size(); ++i)
    {
        for (int j = 0; j < matches[i].size(); j++)
        {
            Point2f from = kpA[matches[i][j].queryIdx].pt;
            Point2f to = kpB[matches[i][j].trainIdx].pt;
            
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
            //calculate local distance for each possible match
            double dist = sqrt((from.x - to.x) * (from.x - to.x) + (from.y - to.y) * (from.y - to.y));
            
            //save as best match if local distance is in specified area and on same height
            if (dist < tresholdDist && abs(from.y-to.y)<5 && (l >= threshold || r >= threshold))
            {
                goodMatches.push_back(matches[i][j]);
                j = (int) matches[i].size();
            }
        }
    }
    
//    for (int i = 0; i < matches.size(); ++i)
//    {
//        const Point2i& pA = kpA[matches[i].queryIdx].pt;
//        const Point2i& pB = kpB[matches[i].trainIdx].pt;
//        
//        int depthA = frameA.getDepth(round(pA.x), round(pA.y));
//        int depthB = frameB.getDepth(round(pB.x), round(pB.y));
//    
//        const Point2f p = pA - pB;
//        
//        float l = cosine(p, left);
//        float r = cosine(p, right);
//        
//        if (l < 0.0f)
//        {
//            l = -l;
//        }
//        
//        if (r < 0.0f)
//        {
//            r = -r;
//        }
//        
//        if (l >= threshold || r >= threshold)
//        {
////            if (depthA > epsilon || depthB > epsilon)
//            {
//                goodMatches.push_back(matches[i]);
//            }
////            else
////            {
////                std::cout << depthA << ", " << depthB << std::endl;
////            }
//        }
//    }
    
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
//    std::cout << matchImage++ << std::endl;
    
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
