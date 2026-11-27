// Shared WebXR bootstrap and WebSocket utilities for the headset and phone frontends
// Provides XR setup, session lifecycle, render loop, and message sending.

import { SESSION_MODE } from '/session-mode.js';
import { WebXRButton } from '/webxr-button.js';

// One controller in a message on `/ws`: its position, its wxyz orientation, the value of each button and the number
// of presses of each button since the page loaded
export function controllerSample(position, orientation, buttons, presses) {
  return { position, orientation, buttons, presses };
}

// The field of a message on `/ws` that carries a WebRTC setup message of the video, from the page or to it
// (`positronic.drivers.webxr`)
const SIGNAL = 'webrtc';

// The WebRTC setup messages on the controller socket. Another channel of setup messages gives the same surface:
// `isOpen`, `send(message)`, and the handlers `onopen` and `onmessage`
export class SocketSignaling {
  constructor() {
    this.socket = null;
    this.onopen = null;
    this.onmessage = null;
  }

  // Carry the setup messages on `socket`, in place of the socket before it
  attach(socket) {
    this.socket = socket;
    socket.onopen = () => this.onopen?.();
    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (SIGNAL in data) this.onmessage?.(data[SIGNAL]);
    };
  }

  get isOpen() {
    return this.socket !== null && this.socket.readyState === WebSocket.OPEN;
  }

  send(message) {
    if (this.isOpen) this.socket.send(JSON.stringify({ [SIGNAL]: message }));
  }
}

export function startXRApp({ onInit, onFrame, onEveryFrame, referenceSpace = 'local', domOverlayRoot = (typeof document !== 'undefined' ? document.body : null) }) {
  let websocket = null;
  const signaling = new SocketSignaling();
  let xrButton = null;
  let xrRefSpace = null;
  let scene = null, camera = null, renderer = null, gl = null;

  function initGL() {
    if (gl) return;
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    const webglCanvas = document.createElement('canvas');
    renderer = new THREE.WebGLRenderer({ canvas: webglCanvas, antialias: true, alpha: true, xrCompatible: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.xr.enabled = true;
    document.body.appendChild(renderer.domElement);
    gl = renderer.getContext();
    const ambientLight = new THREE.AmbientLight(0xffffff, 1.0);
    scene.add(ambientLight);
    if (typeof onInit === 'function') {
      onInit({ scene, camera, renderer, signaling });
    }
  }

  function onRequestSession() {
    const sessionInit = {
      optionalFeatures: ['local-floor', 'dom-overlay']
    };
    if (domOverlayRoot) {
      sessionInit.domOverlay = { root: domOverlayRoot };
    }
    return navigator.xr.requestSession(SESSION_MODE, sessionInit).then((session) => {
      xrButton.setSession(session);
      onSessionStarted(session);
    });
  }

  function onSessionStarted(session) {
    session.addEventListener('end', onSessionEnded);
    initGL();
    renderer.xr.setSession(session);
    try { document.body && document.body.classList.add('xr-active'); } catch {}
    session.requestReferenceSpace(referenceSpace).then((refSpace) => {
      xrRefSpace = refSpace;
      session.requestAnimationFrame(onXRFrame);
    });
  }

  function onSessionEnded(event) {
    // Always reset button state on session end
    xrButton.setSession(null);
    try { document.body && document.body.classList.remove('xr-active'); } catch {}
  }

  function sendControllers(controllers) {
    if (websocket.readyState !== WebSocket.OPEN) return;
    // The timestamp is on the monotonic clock of the page, so that the service can find a message that comes late
    const payload = { controllers, timestamp: performance.now() };
    websocket.send(JSON.stringify(payload));
  }

  function onXRFrame(t, frame) {
    const session = frame.session;
    // Each frame, also one with no viewer pose, so that the page sees each change of its input. `onFrame` gets what it
    // returns
    const everyFrame = typeof onEveryFrame === 'function' ? onEveryFrame({ frame, session, refSpace: xrRefSpace }) : undefined;
    const pose = frame.getViewerPose(xrRefSpace);
    if (pose) {
      const glLayer = session.renderState.baseLayer;
      const view = pose.views[0];
      const viewport = glLayer.getViewport(view);
      gl.bindFramebuffer(gl.FRAMEBUFFER, glLayer.framebuffer);
      gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);
      camera.matrix.fromArray(view.transform.matrix);
      camera.projectionMatrix.fromArray(view.projectionMatrix);
      camera.updateMatrixWorld(true);

      // Allow page to update scene and compute controllers
      let controllers = null;
      if (typeof onFrame === 'function') {
        controllers = onFrame({ frame, session, refSpace: xrRefSpace, view, scene, camera, renderer, everyFrame });
      }

      renderer.render(scene, camera);
      if (controllers) {
        sendControllers(controllers);
      }
    }
    session.requestAnimationFrame(onXRFrame);
  }

  function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    websocket = new WebSocket(`${protocol}//${window.location.host}/ws`);
    signaling.attach(websocket);
    websocket.onclose = () => {
      console.log('Controller WebSocket closed, attempting to reconnect...');
      setTimeout(connectWebSocket, 1000);
    };
  }

  connectWebSocket();

  // Attach button and enable
  xrButton = new WebXRButton({ onRequestSession, onEndSession: (s) => s.end() });
  document.querySelector('header')?.appendChild(xrButton.domElement);
  if (navigator.xr) {
    navigator.xr.isSessionSupported(SESSION_MODE).then((supported) => {
      xrButton.enabled = supported;
    });
  }

  // return handles for optional external use
  return { get scene() { return scene; }, get camera() { return camera; }, get renderer() { return renderer; } };
}
