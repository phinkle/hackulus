Hackulus Thriftus
=================

Kinect
======

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

Displaying with Google Cardboard
================================

We used the Google Cardboard example taken from [Google Chrome Experiments page](http://vr.chromeexperiments.com/example.html).

Mirrored code located at www.cs.utexas.edu/~phinkle/temp-cardboard/cardboard-ply.
This can be run on a mobile device in order to be used with Google Cardboard.

<b>cardboard-ply/ply/</b><br>
Contains all the ply files to display.
Note: .ply files handled by this program do not use the header. Files include only points and colors.

<b>cardboard-ply/textures/</b><br>
Contains images of used for texturing the ground plane.

<b>cardboard-ply/third-party/threejs</b><br>
Contains all the three.js libraries used in main.js.

<b>cardboard-ply/index.html</b><br>
Sets up the canvas to be used to display the WebGL loader.
Also imports all the libraries needed in main.js.

<b>cardboard-ply/main.js</b><br>
Creates a threejs scene using the WebGL loader and offsetting two screens to create a stereo vision view for Google Cardboard.
The ply file is loaded and used to create a point cloud which is then displayed in the threejs scene.

<b>Further improvements</b><br>
- Allow user to change which ply file to view via web interface
- Allow user to move the figure around (currently this is an issue because of limited inputs available to cardboard).
- Calculate the center and size of the point cloud and adjust its placement in relation to the camera dynamically.