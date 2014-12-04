Compilation:
g++ register.cpp -o register `pkg-config --cflags --libs opencv` -O3

Run:
./register icp_iterations sampling_probability

icp_iterations: an positive integer
sampling_probability: double from [0, 1]

Structure:

- Initialize Point Cloud with a random sample of the first Kinect Frame
- Initialize global transformation with 4x4 Identity
- For each next Kinect Frame:
	Extract a random sample from current frame for ICP
	For each ICP iteration:
		- Find correspondences
			- Store points in OpenCV KD-tree
			- Query points to find closest correpondences
				- Not necessarily 1-to-1 mapping
		- SVD on sets of correspondences
		- Include rotation and translation in global transformation
		- Rotate current point cloud
	Register transformed point cloud
	Write point cloud to disk

[JON WRITE THE KINECT PY]
