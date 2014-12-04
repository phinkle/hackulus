'use strict';

var camera, scene, renderer;
var effect, controls;
var element, container;

var clock = new THREE.Clock();

init();
animate();

/**
 * Initialize the scene for threejs by creating a WebGL renderer.
 */
function init() {
  renderer = new THREE.WebGLRenderer();
  renderer.setClearColorHex( 0xa3a3a3, 1 );
  element = renderer.domElement;
  container = document.getElementById('canvas');
  container.appendChild(element);

  // Create the Stereo effect for viewing in Google Cardboard.
  effect = new THREE.StereoEffect(renderer);

  scene = new THREE.Scene();

  // Create a Perspective camera and initialize the position.
  camera = new THREE.PerspectiveCamera(90, 1, 0.001, 700);
  camera.position.set(80, 80, 200);
  scene.add(camera);

  /**
   * Assign orbit controls to the camera.
   */

  controls = new THREE.OrbitControls(camera, element);
  controls.rotateUp(Math.PI / 8);
  controls.panUp(Math.PI / 4);
  controls.target.set(
    camera.position.x + 0.1,
    camera.position.y,
    camera.position.z
  );
  controls.noZoom = true;
  controls.noPan = true;
  controls.autoRotate = false;

  /**
   * To move the object using the orbit controls pass in a THREE.Object3D to the constructor.
   * This assignment causes the view to be stuck looking at the ground on the web browser, but works correctly on mobile.
   * Must connect and update DeviceOrientationControls or the scene will not render correctly.
   */
  controls = new THREE.DeviceOrientationControls(camera, true);
  controls.connect();
  controls.update();

  /**
   * Change orientation on click events as well as on device orientation movement.
   */
  function setOrientationControls(e) {
    if (!e.alpha) {
      return;
    }
    element.addEventListener('click', fullscreen, false);

    window.removeEventListener('deviceorientation', setOrientationControls);
  }
  window.addEventListener('deviceorientation', setOrientationControls, true);

  addLights();
  createGroundPlane();
  displayPoints("ply/batman.ply");

  window.addEventListener('resize', resize, false);
  setTimeout(resize, 1);
}

/**
 * Create a Hemisphere light and a directional light to the scene.
 */
function addLights() {
  var hemiLight = new THREE.HemisphereLight(0xffffff, 0x000000, 0.3);
  scene.add(hemiLight);

  var dirLight = new THREE.DirectionalLight(0xffffff);
  dirLight.position.set(0, 0, 1);
  scene.add(dirLight);
}

/**
 * Add a ground plane to the scene with a checkerboard pattern.
 */
function createGroundPlane() {
  var groundSize = 1000;
  var texture = THREE.ImageUtils.loadTexture(
    'textures/patterns/checker.png'
  );
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat = new THREE.Vector2(50, 50);
  texture.anisotropy = renderer.getMaxAnisotropy();

  var material = new THREE.MeshPhongMaterial({
    color: 0xffffff,
    specular: 0xffffff,
    shininess: 20,
    shading: THREE.FlatShading,
    map: texture
  });

  var geometry = new THREE.PlaneGeometry(groundSize, groundSize);

  var mesh = new THREE.Mesh(geometry, material);
  mesh.rotation.x = -Math.PI / 2;
  scene.add(mesh);
}

/**
 * Takes in a file path of a ply file that will be displayed on the screen.
 */
function displayPoints(file) {
  var rawFile = new XMLHttpRequest();
  rawFile.open("GET", file, false);
  rawFile.onreadystatechange = function () {
      if (rawFile.readyState === 4) {
          if (rawFile.status === 200 || rawFile.status == 0) {
              var allText = rawFile.responseText;
              parsePoints(allText);
          }
      }
  }
  rawFile.send(null);
}

/**
 * Takes in the coordinates and colors of a ply file as a string.
 * Parses through the points and created a point cloud from the data.
 */
function parsePoints(data) {
  var pointSize = 1.0;
  var xOffset = -200;
  var yOffset = 350;
  var zOffset = 0;
  var lines = data.split("\n");
  var geometry = new THREE.Geometry();

  var colors = [];

  for (var i = 0; i < lines.length - 1; i++) {
    var points = lines[i].split(" ");

    /** Threejs/ply coordinate system differences:
     *  x and y are switched.
     *  y is upside down. Multiply by -1 and add a buffer.
     *
     * Note: it is important to convert from string to number here otherwise you will be appending instead of adding.
     */
    var vector = new THREE.Vector3(Number(points[1]) + xOffset, (Number(points[0]) * -1) + yOffset, Number(points[2]) + zOffset);
    geometry.vertices.push(vector);
    var c = new THREE.Color("rgb(" + points[5] + "," + points[4] + "," + points[3] + ")" );
    colors[i] = c;
  }

  geometry.colors = colors;
  geometry.computeBoundingBox();

  var material = new THREE.PointCloudMaterial( { size: pointSize, vertexColors: THREE.VertexColors } );
  var pointcloud = new THREE.PointCloud( geometry, material );
  scene.add(pointcloud);
}

/**
 * Resize the canvas when the screen is resized.
 */
function resize() {
  var width = container.offsetWidth;
  var height = container.offsetHeight;

  camera.aspect = width / height;
  camera.updateProjectionMatrix();

  renderer.setSize(width, height);
  effect.setSize(width, height);
}

/**
 * Update the controls and projection matrix.
 */
function update(dt) {
  resize();

  camera.updateProjectionMatrix();

  controls.update(dt);
}

/**
 * Render the scene using the stereo effect with the current scene and camera.
 */
function render(dt) {
  effect.render(scene, camera);
}

/**
 * Animate the canvas by updating and rendering for each frame.
 */
function animate(t) {
  requestAnimationFrame(animate);

  update(clock.getDelta());
  render(clock.getDelta());
}

/**
 * Handle fullscreen display.
 */
function fullscreen() {
  if (container.requestFullscreen) {
    container.requestFullscreen();
  } else if (container.msRequestFullscreen) {
    container.msRequestFullscreen();
  } else if (container.mozRequestFullScreen) {
    container.mozRequestFullScreen();
  } else if (container.webkitRequestFullscreen) {
    container.webkitRequestFullscreen();
  }
}