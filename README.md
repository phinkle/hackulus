Hackulus Thriftus
=================

Kinect
======

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
