Hackulus Thriftus Google Cardboard Display
==========================================

Google Cardboard example taken from [Google Chrome Experiments page](http://vr.chromeexperiments.com/example.html).

Mirrored code located at www.cs.utexas.edu/~phinkle/temp-cardboard/cardboard-ply
This can be run on a mobile device in order to be used with Google Cardboard.

cardboard-ply/ply/
Contains all the ply files to display.
Note: .ply files handled by this program do not use the header. Files include only points and colors.

cardboard-ply/textures/
Contains images of used for texturing the ground plane.

cardboard-ply/third-party/threejs
Contains all the three.js libraries used in main.js.

cardboard-ply/index.html
Sets up the canvas to be used to display the WebGL loader.
Also imports all the libraries needed in main.js.

cardboard-ply/main.js
Creates a threejs scene using the WebGL loader and offsetting two screens to create a stereo vision view for Google Cardboard.
The ply file is loaded and used to create a point cloud which is then displayed in the threejs scene.

Further improvements: 
- Allow user to change which ply file to view via web interface
- Allow user to move the figure around (currently this is an issue because of limited inputs available to cardboard).
- Calculate the center and size of the point cloud and adjust its placement in relation to the camera dynamically.
