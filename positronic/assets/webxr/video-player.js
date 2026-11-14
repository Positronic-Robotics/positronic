/**
 * WebXR Video Player class that displays a video stream on a plane in AR
 */

// How the plane shows the video: as it is, with black clear so that only bright parts float over the room, or not at all
export const PANEL_MODE = { ON_BLACK: 'on_black', CLEAR_BLACK: 'clear_black', HIDDEN: 'hidden' };
const PANEL_CYCLE = [PANEL_MODE.ON_BLACK, PANEL_MODE.CLEAR_BLACK, PANEL_MODE.HIDDEN];
// The height of the plane in meters, 2 m in front of the eyes. The view with the column of the other cameras is wider
const PANEL_HEIGHT_M = 0.85;

export class WebXRVideoPlayer {
    constructor() {
        this.videoTexture = null;
        this.videoMaterial = null;
        this.videoPlane = null;
        this.videoGroup = null;
        this.ws = null;
        this.decoder = null;
        this.canvas = null;
        this.lastFrameAt = 0;
        this.watchdog = null;
        this.scene = null;
        this.camera = null;
    }

    async init(scene, camera) {
        console.log("Initializing WebXRVideoPlayer");
        this.scene = scene;
        this.camera = camera;

        // The decoder draws each frame on this canvas, and the plane shows the canvas
        this.canvas = document.createElement('canvas');
        this.videoTexture = new THREE.CanvasTexture(this.canvas);
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

        // A plane of a fixed height, whose width follows the aspect of each frame (see showFrame)
        this.videoPlane = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), this.videoMaterial);
        this.videoPlane.scale.set(PANEL_HEIGHT_M * 4 / 3, PANEL_HEIGHT_M, 1);

        // Create a group to hold the video plane
        this.videoGroup = new THREE.Group();
        this.videoGroup.add(this.videoPlane);
        this.scene.add(this.videoGroup);
        this.mode = PANEL_MODE.ON_BLACK;
        this.applyMode();


        // Connect to video websocket
        console.log("Connecting to video websocket...");
        this.connectWebSocket();
        this.watchdog = setInterval(() => this.replaceLostSocket(), 1000);
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/video`;

        this.ws = new WebSocket(wsUrl);
        this.ws.binaryType = 'arraybuffer';
        this.lastFrameAt = performance.now();
        const stream = this.openStream();

        this.ws.onmessage = (event) => {
            this.lastFrameAt = performance.now();
            stream(event.data);
        };

        this.ws.onopen = () => {
            console.log("Video WebSocket connected");
        };

        this.ws.onerror = (error) => {
            console.error("Video WebSocket error:", error);
        };

        this.ws.onclose = () => {
            console.log("Video WebSocket closed");
        };
    }

    // A new H.264 decoder for one socket. Returns the handler of its messages: a JSON decoder configuration as text
    // before each key frame, and each frame as a binary Annex B access unit. A decoder error closes the socket, and
    // the watchdog opens a new one, whose stream starts at a key frame
    openStream() {
        if (this.decoder && this.decoder.state !== 'closed') this.decoder.close();
        const ws = this.ws;
        const decoder = new VideoDecoder({
            output: (frame) => this.showFrame(frame),
            error: (error) => {
                console.error("Video decoder error:", error);
                ws.close();
            },
        });
        this.decoder = decoder;
        let config = null;
        let keyNext = false;
        let timestamp = 0;
        return (data) => {
            if (decoder.state === 'closed') return;
            if (typeof data === 'string') {
                const next = JSON.parse(data);
                if (!config || next.codec !== config.codec || next.width !== config.width || next.height !== config.height) {
                    decoder.configure({
                        codec: next.codec, codedWidth: next.width, codedHeight: next.height, optimizeForLatency: true,
                    });
                    config = next;
                }
                keyNext = true;
                return;
            }
            if (!config) return;
            decoder.decode(new EncodedVideoChunk({ type: keyNext ? 'key' : 'delta', timestamp: timestamp++, data }));
            keyNext = false;
        };
    }

    showFrame(frame) {
        if (this.canvas.width !== frame.displayWidth || this.canvas.height !== frame.displayHeight) {
            this.canvas.width = frame.displayWidth;
            this.canvas.height = frame.displayHeight;
            this.videoTexture.dispose();
        }
        this.canvas.getContext('2d').drawImage(frame, 0, 0);
        frame.close();
        this.videoTexture.needsUpdate = true;
        this.videoPlane.scale.x = PANEL_HEIGHT_M * this.canvas.width / this.canvas.height;
    }

    replaceLostSocket() {
        // The service sends a frame on every view tick. A socket with no frame for this long is lost: the page never
        // sends on it, so a connection that dies with no close frame never closes on the page.
        const frameTimeoutMs = 3000;
        const silentMs = performance.now() - this.lastFrameAt;
        if (this.ws.readyState !== WebSocket.CLOSED && silentMs < frameTimeoutMs) return;
        console.log(`Video WebSocket lost (no frame for ${Math.round(silentMs)} ms), reconnecting...`);
        this.ws.close();
        this.connectWebSocket();
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

    // The next mode in PANEL_CYCLE: on black, with black clear, hidden, and on black again. Returns the new mode
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
        clearInterval(this.watchdog);
        if (this.ws) {
            this.ws.close();
        }
        if (this.decoder && this.decoder.state !== 'closed') {
            this.decoder.close();
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
