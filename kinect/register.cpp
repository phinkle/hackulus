//
//  register.cpp
//  Use ICP to register a set of 3D point clouds.
//
//  Created by Jonathan Lee on 11/30/14.
//  Copyright (c) 2014 Jonathan Lee. All rights reserved.
//
// g++ register.cpp -o register `pkg-config --cflags --libs opencv` -O3

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <utility>
<<<<<<< HEAD
#include <ctime>
=======
#include <queue>
#include <cmath>
#include <time.h>
>>>>>>> 783ae64cd6bae0278b46a442c7da5ab6d774faae

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv/highgui.h>

using namespace cv;

// p should be a nx6 which represents the point cloud
// each row is a single point: (x, y, z, b, g, r)
Mat applyTransformation(const Mat &p, const Mat &r, const Mat &t) {
    Mat c = p.colRange(0, 3);
    Mat c_ = r * c.t();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < c_.row(i).cols; ++j) {
            c_.row(i).col(j) += t.at<float>(i);
        }
    }
    c_ = c_.t();
    Mat p_ = p.clone();
    for (int i = 0; i < p_.rows; ++i) {
        for (int j = 0; j < 3; ++j) {
            p_.at<float>(i, j) = c_.at<float>(i, j);
        }
    }
    return p_;
}

// Takes a matrix a and b, which are both mx3 matrices
// the cooresponding rows in each matrix represent the matching
// pairs.
//
// "Returns" the 3x3 rotation matrix and the 3x1 translation vector
// to transform a onto b
void rigid_transform_3D(const Mat &a, const Mat &b, Mat &r, Mat &t) {
    // Total number of point pairs
    int n = a.rows;
    
    // Find the centroids
    Mat centroid_a = Mat::zeros(1, 3, CV_32F);
    Mat centroid_b = Mat::zeros(1, 3, CV_32F);
    for (int i = 0; i < n; ++i) {
        centroid_a.row(0) += a.row(i);
        centroid_b.row(0) += b.row(i);
    }
    centroid_a /= n;
    centroid_b /= n;
    
    // Center the points
    Mat aa = a.clone();
    Mat bb = b.clone();
    for (int i = 0; i < n; ++i) {
        aa.row(i) -= centroid_a;
        bb.row(i) -= centroid_b;
    }
    
    Mat u, w, vt;
    SVD::compute(aa.t() * bb, w, u, vt);
    
    r = vt.t() * u.t();
    if (determinant(r) < 0) {
        vt.row(2) *= -1;
        r = vt.t() * u.t();
    }
    
    t = -r * centroid_a.t() + centroid_b.t();
}

// Takes a matrix pc_a and pc_b which are both nx6 matrices
// where the number of rows represents the number of
// point clouds and the cols represent the (x, y, z, b, g, r)
// of each point
//
// "Returns" matrices a and b, which are both mx3 matrices
// the zip of the two matrices represents the (x, y, z) of the
// points that match
void nearest_neighbors(flann::Index& kdtree, const Mat &pc_a, const Mat &pc_b, Mat &a, Mat &b) {
    a = Mat::zeros(pc_b.rows, 3, CV_32F);
    b = pc_b.colRange(0, 3).clone();
    Mat indices = Mat::zeros(1, 1, CV_32F);
    Mat dists = Mat::zeros(1, 1, CV_32F);
    
    for (int i = 0; i < b.rows; ++i) {
        kdtree.knnSearch(b.row(i), indices, dists, 1, flann::SearchParams(64));
        pc_a.row(indices.at<int>(0)).colRange(0, 3).copyTo(a.row(i));
    }
}

Mat load_kinect_frame(const string image_filename, const string depth_filename) {
    Mat image, pc;
    int dim_row, dim_col;
    
    image = imread(image_filename);
    if (!image.data) {
        std::cout << "Could not read '" << image_filename << "'\n";
        throw Exception();
    }
    dim_row = image.rows;
    dim_col = image.cols;
    image = image.reshape(3, image.rows * image.cols);
    
    std::ifstream file(depth_filename);
    pc = Mat::zeros(1, 6, CV_32F);
    pc.pop_back(1);
    string line;
    for (int i = 0; getline(file, line); ++i) {
        std::istringstream in(line);
        float x, y, z, b, g, r;
        in >> z;
        if (z > 0) {
            x = i / dim_col;
            y = i % dim_col;
            b = image.at<Vec3b>(i, 0)[0];
            g = image.at<Vec3b>(i, 0)[1];
            r = image.at<Vec3b>(i, 0)[2];
            Mat p = (Mat_<float>(1, 6) << x, y, z, b, g, r);
            pc.push_back(p);
        }
    }
    return pc;
}

void save_point_cloud(Mat &a, string filename) {
    std::fstream plyFile;
    
    plyFile.open(filename, std::fstream::out);
    plyFile << "ply" << std::endl;
    plyFile << "format ascii 1.0" << std::endl;
    plyFile << "element vertex " << a.rows << std::endl;
    plyFile << "property float x" << std::endl;
    plyFile << "property float y" << std::endl;
    plyFile << "property float z" << std::endl;
    plyFile << "property uchar blue" << std::endl;
    plyFile << "property uchar green" << std::endl;
    plyFile << "property uchar red" << std::endl;
    plyFile << "end_header" << std::endl;
    
    for (int i = 0; i < a.rows; ++i) {
        for (int j = 0; j < 3; ++j) {
            plyFile << a.at<float>(i, j) << " ";
        }
        for (int j = 3; j < 5; ++j) {
            plyFile << (int) round(a.at<float>(i, j)) << " ";
        }
        plyFile << (int) round(a.at<float>(i, 5)) << "\n";
    }
    
    plyFile.close();
}

<<<<<<< HEAD
Mat get_rotation_translation_matrix(const Mat& rotation, const Mat& translation)
{
    Mat matrix = Mat::eye(4, 4, CV_32F);

=======
void extractRigidTransform(const Mat& m, Mat& rotation, Mat& translation)
{
    rotation = Mat::zeros(3, 3, CV_32F);
    translation = Mat::zeros(3, 1, CV_32F);
    
>>>>>>> 783ae64cd6bae0278b46a442c7da5ab6d774faae
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
<<<<<<< HEAD
            matrix.at<float>(i, j) = rotation.at<float>(i, j);
        }
    }

    matrix.at<float>(0, 3) = translation.at<float>(0, 0);
    matrix.at<float>(1, 3) = translation.at<float>(0, 1);
    matrix.at<float>(2, 3) = translation.at<float>(0, 2);

    return matrix;
}

void get_rotation_translation_from_matrix(const Mat& matrix, Mat& rotation, Mat& translation)
{
    rotation = Mat::zeros(3, 3, CV_32F);
    translation = Mat::zeros(1, 3, CV_32F);

=======
            rotation.at<float>(i, j) = m.at<float>(i, j);
        }
    }
    
    translation.at<float>(0, 0) = m.at<float>(0, 3);
    translation.at<float>(0, 1) = m.at<float>(1, 3);
    translation.at<float>(0, 2) = m.at<float>(2, 3);
}

Mat getRigidTransform(const Mat& rotation, const Mat& translation)
{
    Mat m = Mat::eye(4, 4, CV_32F);
    
>>>>>>> 783ae64cd6bae0278b46a442c7da5ab6d774faae
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
<<<<<<< HEAD
            rotation.at<float>(i, j) = matrix.at<float>(i, j);
        }
    }

    translation.at<float>(0, 0) = matrix.at<float>(0, 3);
    translation.at<float>(0, 1) = matrix.at<float>(1, 3);
    translation.at<float>(0, 2) = matrix.at<float>(2, 3);
=======
            m.at<float>(i, j) = rotation.at<float>(i, j);
        }
    }
    
    m.at<float>(0, 3) = translation.at<float>(0, 0);
    m.at<float>(1, 3) = translation.at<float>(0, 1);
    m.at<float>(2, 3) = translation.at<float>(0, 2);
    
    return m;
}

void applyRotationsAndTranslations(Mat& m, const vector<Mat>& rotations, const vector<Mat>& translations)
{
    assert(rotations.size() == translations.size());
    
    for (int i = 0; i < rotations.size(); ++i) {
        m = applyTransformation(m, rotations[i], translations[i]);
    }
>>>>>>> 783ae64cd6bae0278b46a442c7da5ab6d774faae
}

int main(int argc, char **argv) {
    if (argc < 3)
        return 1;
    
    string pc_filepath = argv[1];
    string pc_file_out_ply = argv[2];
    int icp_num_iters = std::atoi(argv[3]);
    pc_filepath += "/";
    
    std::ifstream file(pc_filepath + "config.txt");
    string line;
    getline(file, line);
    std::istringstream in(line);
    
    int num_images;
    in >> num_images;
    
    Mat pc_a = load_kinect_frame(pc_filepath + "image_0.png",
                                 pc_filepath + "depth_0.txt");
<<<<<<< HEAD

    Mat transformation = Mat::eye(4, 4, CV_32F);
    Mat r, t;
    clock_t time;

=======
    
    Mat transformation = Mat::eye(4, 4, CV_32F);
    Mat rotation, translation;
    clock_t time;
    
>>>>>>> 783ae64cd6bae0278b46a442c7da5ab6d774faae
    for (int image_num = 1; image_num < num_images; ++image_num) {
        // step 1 read in two point clouds
        std::cout << "REGISTERING IMAGE " << image_num << "\n";
        std::cout << "Step 1: ";
        time = clock();
        string str_num = std::to_string(image_num);
        Mat pc_b = load_kinect_frame(pc_filepath + "image_" + str_num + ".png",
                                     pc_filepath + "depth_" + str_num + ".txt");
<<<<<<< HEAD

        get_rotation_translation_from_matrix(transformation, r, t);
        applyTransformation(pc_b, r, t);
        std::cout << "complete\n";

=======
        std::cout << "complete " << ((float)(clock() - time)) / CLOCKS_PER_SEC << std::endl;
        
        extractRigidTransform(transformation, rotation, translation);
        pc_b = applyTransformation(pc_b, rotation, translation);
        
>>>>>>> 783ae64cd6bae0278b46a442c7da5ab6d774faae
        // step 2 call nearest_neighbors
        for (int i = 0; i < icp_num_iters; ++i) {
            flann::Index kdtree(pc_a.colRange(0, 3).clone(), flann::KDTreeIndexParams(1));
            std::cout << "Step 2: ";
            time = clock();
            Mat a, b;
<<<<<<< HEAD
            time = clock();
            nearest_neighbors(pc_a, pc_b, a, b);
            time = clock() - time;
            std::cout << "Completed in " << ((float)time) / CLOCKS_PER_SEC) << " seconds" << std::endl;


            // step 3 pass into rigid transform
            std::cout << "Step 3: ";
            time = clock();
            rigid_transform_3D(a, b, r, t);
            time = clock() - time;
            std::cout << "Completed in " << ((float)time) / CLOCKS_PER_SEC) << " seconds" << std::endl;

=======
            nearest_neighbors(kdtree, pc_a, pc_b, a, b);
            std::cout << "complete " << ((float)(clock() - time)) / CLOCKS_PER_SEC << std::endl;
            
            
            // step 3 pass into rigid transform
            std::cout << "Step 3: ";
            time = clock();
            Mat r, t;
            rigid_transform_3D(b, a, r, t);
            std::cout << "complete " << ((float)(clock() - time)) / CLOCKS_PER_SEC << std::endl;
            
            
>>>>>>> 783ae64cd6bae0278b46a442c7da5ab6d774faae
            // step 4 apply transformation to second point cloud
            std::cout << "Step 4: ";
            time = clock();
            pc_b = applyTransformation(pc_b, r, t);
<<<<<<< HEAD
            Mat newTransform = get_rotation_translation_matrix(r, t);
            transformation *= newTransform;
            time = clock() - time;
            std::cout << "Completed in " << ((float)time) / CLOCKS_PER_SEC) << " seconds" << std::endl;
=======
            transformation *= getRigidTransform(r, t);
            std::cout << "complete " << ((float)(clock() - time)) / CLOCKS_PER_SEC << std::endl;
>>>>>>> 783ae64cd6bae0278b46a442c7da5ab6d774faae
        }
        
        // step 5 repeat 2 - 5 as many times as needed
        // step 6 combine two point clouds and output to ply file
        std::cout << "Step 5: ";
        time = clock();
        Mat combined;
        vconcat(pc_a, pc_b, combined);
        pc_a = combined;
        save_point_cloud(pc_a, pc_file_out_ply);
<<<<<<< HEAD
        time = clock() - time;
        std::cout << "Completed in " << ((float)time) / CLOCKS_PER_SEC) << " seconds" << std::endl;
=======
        std::cout << "complete " << ((float)(clock() - time)) / CLOCKS_PER_SEC << std::endl;
>>>>>>> 783ae64cd6bae0278b46a442c7da5ab6d774faae
    }
    
    return 0;
}
