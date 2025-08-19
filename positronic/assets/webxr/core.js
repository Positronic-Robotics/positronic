// Shared WebXR bootstrap and WebSocket utilities for Oculus/iPhone frontends
// Provides XR setup, session lifecycle, render loop, and message sending.

import { WebXRButton } from '/webxr-button.js';

export function initWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws`;
  const ws = new WebSocket(wsUrl);
  return ws;
}

export function sendControllers(ws, controllers) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const payload = { controllers, timestamp: Date.now() };
  ws.send(JSON.stringify(payload));
}

export function startXRApp({ websocket, onInit, onFrame, referenceSpace = 'local', domOverlayRoot = (typeof document !== 'undefined' ? document.body : null) }) {
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
      onInit({ scene, camera, renderer });
    }
  }

  function onRequestSession() {
    const sessionInit = {
      optionalFeatures: ['local-floor', 'dom-overlay']
    };
    if (domOverlayRoot) {
      sessionInit.domOverlay = { root: domOverlayRoot };
    }
    return navigator.xr.requestSession('immersive-ar', sessionInit).then((session) => {
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

  function onXRFrame(t, frame) {
    const session = frame.session;
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
        controllers = onFrame({ frame, session, refSpace: xrRefSpace, view, scene, camera, renderer });
      }

      renderer.render(scene, camera);
      if (controllers) {
        sendControllers(websocket, controllers);
      }
    }
    session.requestAnimationFrame(onXRFrame);
  }

  // Attach button and enable
  xrButton = new WebXRButton({ onRequestSession, onEndSession: (s) => s.end() });
  document.querySelector('header')?.appendChild(xrButton.domElement);
  if (navigator.xr) {
    navigator.xr.isSessionSupported('immersive-ar').then((supported) => {
      xrButton.enabled = supported;
    });
  }

  // return handles for optional external use
  return { get scene() { return scene; }, get camera() { return camera; }, get renderer() { return renderer; } };
}
