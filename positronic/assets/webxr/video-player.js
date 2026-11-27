/**
 * WebXR Video Player class that displays a video stream on a plane in AR
 */

import { WebRTCVideo } from './webrtc-video.js';

// How the plane shows the video: as it is, as it is for a view with no task, with black clear so that only bright parts
// float over the room, or not at all
export const PANEL_MODE = { ON_BLACK: 'on_black', NO_TASK: 'no_task', CLEAR_BLACK: 'clear_black', HIDDEN: 'hidden' };
const PANEL_CYCLE = [PANEL_MODE.ON_BLACK, PANEL_MODE.NO_TASK, PANEL_MODE.CLEAR_BLACK, PANEL_MODE.HIDDEN];
// The height of the plane in meters, 2 m in front of the eyes. The view with the column of the other cameras is wider
const PANEL_HEIGHT_M = 0.85;

export class WebXRVideoPlayer {
    constructor() {
        this.videoTexture = null;
        this.videoMaterial = null;
        this.videoPlane = null;
        this.videoGroup = null;
        this.video = null;
        this.rtc = null;
        this.scene = null;
        this.camera = null;
    }

    // `signaling` carries the WebRTC setup messages of the video (`SocketSignaling` of core.js)
    async init(scene, camera, signaling) {
        console.log("Initializing WebXRVideoPlayer");
        this.scene = scene;
        this.camera = camera;

        // The WebRTC track plays in this element, and the plane shows it
        this.video = document.createElement('video');
        this.video.muted = true;
        this.video.playsInline = true;
        this.videoTexture = new THREE.VideoTexture(this.video);
        this.videoMaterial = new THREE.ShaderMaterial({
            uniforms: {
                map: { value: this.videoTexture },
                blueBoost: { value: 1.0 },
                contrast: { value: 1.0 },  // 1.0 is normal, >1.0 increases contrast
                // 1.0 shows black as clear: text on black floats over the room
                clearBlack: { value: 0.0 }
            },
            vertexShader: `
                varying vec2 vUv;
                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform sampler2D map;
                uniform float blueBoost;
                uniform float contrast;
                uniform float clearBlack;
                varying vec2 vUv;
                void main() {
                    vec4 texColor = texture2D(map, vUv);
                    // Apply contrast adjustment
                    texColor.rgb = (texColor.rgb - 0.5) * contrast + 0.5;
                    // Boost blue channel
                    texColor.b = min(1.0, texColor.b * blueBoost);
                    // Clamp final colors to valid range
                    texColor.rgb = clamp(texColor.rgb, 0.0, 1.0);
                    float brightest = max(texColor.r, max(texColor.g, texColor.b));
                    if (clearBlack > 0.5 && brightest < 0.15) {
                        texColor.a = 0.0;
                    }
                    gl_FragColor = texColor;
                }
            `,
            side: THREE.DoubleSide,
            transparent: true,
        });

        // A plane of a fixed height, whose width follows the aspect of the video (see fitPlane)
        this.videoPlane = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), this.videoMaterial);
        this.videoPlane.scale.set(PANEL_HEIGHT_M * 4 / 3, PANEL_HEIGHT_M, 1);

        // Create a group to hold the video plane
        this.videoGroup = new THREE.Group();
        this.videoGroup.add(this.videoPlane);
        this.scene.add(this.videoGroup);
        this.mode = PANEL_MODE.ON_BLACK;
        this.applyMode();

        this.video.addEventListener('resize', () => this.fitPlane());
        this.rtc = new WebRTCVideo(signaling, this.video);
    }

    fitPlane() {
        if (this.video.videoHeight === 0) return;
        this.videoPlane.scale.x = PANEL_HEIGHT_M * this.video.videoWidth / this.video.videoHeight;
    }

    update() {
        if (!this.videoGroup || !this.camera || !this.videoGroup.visible) return;

        // Get camera position and orientation
        const cameraPosition = new THREE.Vector3();
        const cameraQuaternion = new THREE.Quaternion();
        this.camera.getWorldPosition(cameraPosition);
        this.camera.getWorldQuaternion(cameraQuaternion);

        // Position the video plane 2 meters in front of the camera
        const offset = new THREE.Vector3(0, 0, -2);
        offset.applyQuaternion(cameraQuaternion);

        // Update video plane position and orientation
        this.videoGroup.position.copy(cameraPosition).add(offset);
        this.videoGroup.quaternion.copy(cameraQuaternion);
    }

    applyMode() {
        if (!this.videoGroup) return;
        this.videoGroup.visible = this.mode !== PANEL_MODE.HIDDEN;
        this.videoMaterial.uniforms.clearBlack.value = this.mode === PANEL_MODE.CLEAR_BLACK ? 1.0 : 0.0;
    }

    // The next mode in PANEL_CYCLE: on black, on black with no task, with black clear, hidden, and on black again.
    // Returns the new mode
    cycleMode() {
        this.mode = PANEL_CYCLE[(PANEL_CYCLE.indexOf(this.mode) + 1) % PANEL_CYCLE.length];
        this.applyMode();
        return this.mode;
    }

    show() {
        if (this.videoGroup) {
            console.log("Showing video plane");
            this.videoGroup.visible = true;
        }
    }

    hide() {
        if (this.videoGroup) {
            console.log("Hiding video plane");
            this.videoGroup.visible = false;
        }
    }

    dispose() {
        if (this.rtc) {
            this.rtc.close();
        }
        if (this.videoTexture) {
            this.videoTexture.dispose();
        }
        if (this.videoMaterial) {
            this.videoMaterial.dispose();
        }
        if (this.videoPlane) {
            this.videoPlane.geometry.dispose();
        }
        if (this.videoGroup && this.scene) {
            this.scene.remove(this.videoGroup);
        }
    }
}
