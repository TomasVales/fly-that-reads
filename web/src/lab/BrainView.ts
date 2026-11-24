import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

/** las 127 mil neuronas del cerebro del laboratorio, encendidas con su actividad real */
export class BrainView {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private points: THREE.Points;
  private act: Float32Array;
  private target: Float32Array;
  private off: Float32Array;
  private raf = 0;
  private ro: ResizeObserver;
  private last = performance.now();
  dreaming = 0;
  private idle = 0;

  constructor(private host: HTMLElement, pos: Int16Array, flags: Uint8Array) {
    const N = flags.length;
    this.renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    host.appendChild(this.renderer.domElement);
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.01, 50);
    this.camera.position.set(0, 0.25, 3.1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.enablePan = false;
    this.controls.autoRotate = true;
    this.controls.autoRotateSpeed = 0.6;
    this.controls.minDistance = 1.2;
    this.controls.maxDistance = 6;
    this.renderer.domElement.addEventListener("pointerdown", () => (this.controls.autoRotate = false), { once: true });

    const p = new Float32Array(N * 3);
    for (let i = 0; i < N * 3; i++) p[i] = pos[i] / 32767;
    const kind = new Float32Array(N);
    for (let i = 0; i < N; i++) kind[i] = flags[i] & 2 ? 2 : flags[i] & 1 ? 1 : 0;
    this.act = new Float32Array(N);
    this.target = new Float32Array(N);
    this.off = new Float32Array(N);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(p, 3));
    g.setAttribute("kind", new THREE.BufferAttribute(kind, 1));
    g.setAttribute("act", new THREE.BufferAttribute(this.act, 1).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute("off", new THREE.BufferAttribute(this.off, 1).setUsage(THREE.DynamicDrawUsage));
    const m = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: { px: { value: this.renderer.getPixelRatio() }, dream: { value: 0 } },
      vertexShader: `
        attribute float kind; attribute float act; attribute float off;
        uniform float px; uniform float dream;
        varying vec3 vC; varying float vA;
        void main() {
          vec3 base = kind > 1.5 ? vec3(1.0, 0.72, 0.32) : kind > 0.5 ? vec3(0.45, 0.72, 0.95) : vec3(0.82, 0.84, 0.9);
          base = mix(base, vec3(0.62, 0.5, 1.0), dream * (kind < 0.5 ? 0.8 : 0.3));
          float a = clamp(act, 0.0, 1.0);
          vC = off > 0.5 ? vec3(0.9, 0.18, 0.15) : mix(base, vec3(1.0), a * 0.55);
          vA = off > 0.5 ? 0.16 : (kind > 0.5 ? 0.16 : 0.05) + a * 0.9;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = px * (1.1 + a * 2.6) * (2.6 / -mv.z);
          gl_Position = projectionMatrix * mv;
        }`,
      fragmentShader: `
        varying vec3 vC; varying float vA;
        void main() {
          vec2 d = gl_PointCoord - 0.5; float r = dot(d, d);
          if (r > 0.25) discard;
          gl_FragColor = vec4(vC, vA * (1.0 - r * 3.2));
        }`,
    });
    this.points = new THREE.Points(g, m);
    this.points.rotation.x = 0.35;
    this.scene.add(this.points);

    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(host);
    this.resize();
    const loop = (now: number) => {
      this.raf = requestAnimationFrame(loop);
      const dt = Math.min(0.1, (now - this.last) / 1000);
      this.last = now;
      // la actividad se acerca rápido a la medida y se apaga despacio (como un brillo que queda)
      if (this.idle++ < 120) {
        const up = 1 - Math.exp(-dt * 30), down = 1 - Math.exp(-dt * 3);
        for (let i = 0; i < N; i++) {
          const t = this.target[i], a = this.act[i];
          this.act[i] = a + (t - a) * (t > a ? up : down);
        }
        (g.attributes.act as THREE.BufferAttribute).needsUpdate = true;
      }
      const u = m.uniforms.dream;
      u.value += (this.dreaming - u.value) * Math.min(1, dt * 2);
      this.controls.autoRotateSpeed = this.dreaming ? 0.25 : 0.6;
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    };
    this.raf = requestAnimationFrame(loop);
  }

  /** actividad de una letra (0–255 por neurona); se ve más el cambio que el nivel de fondo */
  setActivity(a: Uint8Array) {
    for (let i = 0; i < a.length; i++) this.target[i] = Math.pow(a[i] / 255, 1.6) * 1.4;
    this.idle = 0;
  }

  setLesioned(flags: Uint8Array, bits: number) {
    for (let i = 0; i < flags.length; i++) this.off[i] = flags[i] & bits ? 1 : 0;
    (this.points.geometry.attributes.off as THREE.BufferAttribute).needsUpdate = true;
  }

  private resize() {
    const w = this.host.clientWidth, h = this.host.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = "100%";
    this.renderer.domElement.style.height = "100%";
    this.camera.aspect = w / h;
    this.camera.position.setLength(Math.max(2.75, (2.75 * 1.25 * h) / w)); // columnas angostas: alejar para que entre el cerebro
    this.camera.updateProjectionMatrix();
  }

  dispose() {
    cancelAnimationFrame(this.raf);
    this.ro.disconnect();
    this.controls.dispose();
    this.points.geometry.dispose();
    (this.points.material as THREE.Material).dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
