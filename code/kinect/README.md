Files:

- kinect.py: captures images from Kinect
- main.cpp: main registration code
- main-tests.cpp: tests for icp.cpp
- icp.cpp/hpp: our library for performing ICP
- Makefile

Compilation:
make

Run:
./kinect.py output_directory
- output_directory: directory to output the color and depth information
./register input_directory output.ply icp_iterations sampling_probability
- input_directory: directory to read color and depth information from
- output.ply: where to output the registered point cloud
- icp_iterations: an positive integer
- sampling_probability: double from [0, 1]
./register-tests

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

To use the kinect.py script, first execute the script. Press ESC to
start recording. Press ESC again to stop. W and S change the scope
of the depth that is recorded. You can see the result live in the
"bgr camera - filtered" window.
