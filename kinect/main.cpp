#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <utility>
#include <queue>
#include <cmath>
#include <time.h>
#include <cstdlib>

#include "icp.hpp"

int main(int argc, char **argv) {
  if (argc < 5) {
    std::cout << "Usage: "
              << argv[0]
              << " path_to_scans/ output.ply icp_iters subsample_probability" << std::endl;
    return 1;
  }

  // Command line parameters
  string pc_filepath = argv[1];
  string pc_file_out_ply = argv[2];
  int icp_num_iters = std::atoi(argv[3]);
  double probability = std::atof(argv[4]);
  if (pc_filepath.c_str()[pc_filepath.length() - 1] != '/') {
    pc_filepath += "/";
  }

  std::ifstream file(pc_filepath + "config.txt");
  string line;
  getline(file, line);
  std::istringstream in(line);

  int num_images;
  in >> num_images;

  Mat pc_a = load_kinect_frame(pc_filepath + "image_0.png",
      pc_filepath + "depth_0.txt");

  Mat allSamples = selectRandomPoints(pc_a, probability);

  // Used for accumulating the rigid transformation matrix
  Mat transformation = Mat::eye(4, 4, CV_32F);

  Mat rotation, translation;
  clock_t time;

  for (int image_num = 1; image_num < num_images; ++image_num) {
    std::cout << "REGISTERING IMAGE " << image_num << std::endl;
    time = clock();

    // Load the point cloud to be registered
    string str_num = std::to_string(image_num);
    Mat pc_b = load_kinect_frame(pc_filepath + "image_" + str_num + ".png",
        pc_filepath + "depth_" + str_num + ".txt");
    Mat pc_b_sample = selectRandomPoints(pc_b, probability);

    // Apply the previous transformations to b so that it is positioned near
    // the last accumulated points
    extractRigidTransform(transformation, rotation, translation);
    pc_b_sample = applyTransformation(pc_b_sample, rotation, translation);
    pc_b = applyTransformation(pc_b, rotation, translation);

    // Perform the specified number of icp iterations
    for (int i = 0; i < icp_num_iters; ++i) {
      // Find the nearest neighbor pairs in the two point clouds
      Mat a, b;
      nearest_neighbors(allSamples, pc_b_sample, a, b);

      // Find the optimal rotation and translation matrix to transform b onto a
      Mat r, t;
      rigid_transform_3D(b, a, r, t);

      // Apply the rigid transformation to the b point cloud
      pc_b_sample = applyTransformation(pc_b_sample, r, t);
      pc_b = applyTransformation(pc_b, r, t);
      transformation *= getRigidTransform(r, t);
    }

    // Combine the two point clouds and save to disk
    Mat combined, combinedSample;
    vconcat(pc_a, pc_b, combined);
    vconcat(allSamples, pc_b_sample, combinedSample);
    pc_a = combined;
    allSamples = combinedSample;
    save_point_cloud(combined, pc_file_out_ply);

    std::cout << "complete " << ((float)(clock() - time)) / CLOCKS_PER_SEC << std::endl;
  }

  return 0;
}
