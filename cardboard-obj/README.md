Cardboard Example
=================

Google Cardboard example, using device orientation API.

Taken from [here](http://vr.chromeexperiments.com/example.html)

Rotating an Object:
http://stackoverflow.com/questions/26081699/threejs-rotating-object-with-device-orientation-control

First Person Controls:
this.controls.disconnect();

if (has.mobile) {
  this.controls = new THREE.DeviceOrientationControls(this.cameras.firstPerson);
} else {
  this.controls = new THREE.FirstPersonControls(this.cameras.firstPerson);
}

this.controls.connect();

this.controls.movementSpeed = 10;
this.controls.rollSpeed = Math.PI / 4;
this.controls.autoForward = true;
