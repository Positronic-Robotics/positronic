/**
 * WebXR Video Player class that displays a video stream on a plane in AR
 */
export class WebXRVideoPlayer {
    constructor() {
        this.videoTexture = null;
        this.videoMaterial = null;
        this.videoPlane = null;
        this.videoGroup = null;
        this.ws = null;
        this.scene = null;
        this.camera = null;
    }

    async init(scene, camera) {
        console.log("Initializing WebXRVideoPlayer");
        this.scene = scene;
        this.camera = camera;

        // Create video texture and material
        this.videoTexture = new THREE.Texture();
        this.videoMaterial = new THREE.ShaderMaterial({
            uniforms: {
                map: { value: this.videoTexture },
                blueBoost: { value: 1.5 },
                contrast: { value: 2.0 }  // 1.0 is normal, >1.0 increases contrast
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
                varying vec2 vUv;
                void main() {
                    vec4 texColor = texture2D(map, vUv);
                    // Apply contrast adjustment
                    texColor.rgb = (texColor.rgb - 0.5) * contrast + 0.5;
                    // Boost blue channel
                    texColor.b = min(1.0, texColor.b * blueBoost);
                    // Clamp final colors to valid range
                    texColor.rgb = clamp(texColor.rgb, 0.0, 1.0);
                    gl_FragColor = texColor;
                }
            `,
            side: THREE.DoubleSide,
            transparent: true,
        });

        // Create plane for video (1 meter wide, 16:9 aspect ratio)
        const width = 1.0;
        const height = width * (9/16);
        this.videoPlane = new THREE.Mesh(
            new THREE.PlaneGeometry(width, height),
            this.videoMaterial
        );

        // this.videoPlane.rotation.z = -Math.PI / 2;

        // Create a group to hold the video plane
        this.videoGroup = new THREE.Group();
        this.videoGroup.add(this.videoPlane);
        this.scene.add(this.videoGroup);
        this.videoGroup.visible = true;


        // Connect to video websocket
        console.log("Connecting to video websocket...");
        await this.connectWebSocket();
    }

    async connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/video`;

        this.ws = new WebSocket(wsUrl);

        this.ws.onmessage = async (event) => {
            try {
                // Create an image from the base64 data
                const img = new Image();
                img.onload = () => {
                    this.videoTexture.image = img;
                    this.videoTexture.needsUpdate = true;
                };
                img.src = 'data:image/jpeg;base64,' + event.data;
            } catch (error) {
                console.error("Error processing video frame:", error);
            }
        };

        this.ws.onopen = () => {
            console.log("Video WebSocket connected");
        };

        this.ws.onerror = (error) => {
            console.error("Video WebSocket error:", error);
        };

        this.ws.onclose = () => {
            console.log("Video WebSocket closed, attempting to reconnect...");
            setTimeout(() => this.connectWebSocket(), 1000);
        };
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
        if (this.ws) {
            this.ws.close();
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
