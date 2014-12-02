'use strict';

var camera, scene, renderer;
var effect, controls;
var element, container;

var clock = new THREE.Clock();

init();
animate();

function init() {
  renderer = new THREE.WebGLRenderer();
  renderer.setClearColorHex( 0xa3a3a3, 1 );
  element = renderer.domElement;
  container = document.getElementById('example');
  container.appendChild(element);

  effect = new THREE.StereoEffect(renderer);

  scene = new THREE.Scene();

  camera = new THREE.PerspectiveCamera(90, 1, 0.001, 700);
  camera.position.set(0, 10, 0);
  scene.add(camera);

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

  function setOrientationControls(e) {
    if (!e.alpha) {
      return;
    }

    element.addEventListener('click', fullscreen, false);

    window.removeEventListener('deviceorientation', setOrientationControls);
  }
  window.addEventListener('deviceorientation', setOrientationControls, true);


  var light = new THREE.HemisphereLight(0xffffff, 0x000000, 0.3);
  scene.add(light);

  var dirLight = new THREE.DirectionalLight(0xffffff);
  dirLight.position.set( 0, 0, 1);
  scene.add(dirLight);

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

  var geometry = new THREE.PlaneGeometry(1000, 1000);

  // adding ground plane
  var m = new THREE.Mesh(geometry, material);
  m.rotation.x = -Math.PI / 2;
  scene.add(m);

  var mesh;

  // loading ply file
  var loader = new THREE.PLYLoader();
        loader.addEventListener( 'load', function ( event ) {
          console.log("found ply file");

          var geometry = event.content;
          var material = new THREE.MeshPhongMaterial( { ambient: 0x0055ff, color: 0x0055ff, specular: 0x111111, shininess: 200 } );
          mesh = new THREE.Mesh( geometry, material );
          controls = new THREE.DeviceOrientationControls(mesh, true);

          mesh.position.set( 30, 20, 5);
          mesh.rotation.set( 0, - Math.PI / 2, 0 );
          mesh.scale.set( 0.05, 0.05, 0.05 );

          scene.add(mesh);

        } );
        loader.load( 'ply/dolphins.ply' );



  window.addEventListener('resize', resize, false);
  setTimeout(resize, 1);
}

function resize() {
  var width = container.offsetWidth;
  var height = container.offsetHeight;

  camera.aspect = width / height;
  camera.updateProjectionMatrix();

  renderer.setSize(width, height);
  effect.setSize(width, height);
}

function update(dt) {
  resize();

  camera.updateProjectionMatrix();

  controls.update(dt);
}

function render(dt) {
  effect.render(scene, camera);
}

function animate(t) {
  requestAnimationFrame(animate);

  update(clock.getDelta());
  render(clock.getDelta());
}

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