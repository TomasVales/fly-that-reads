import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";

/**
 * Escena 3D: una mosca (hecha por código) leyendo un libro abierto sobre un escritorio,
 * con su cerebro (nube de puntos con posiciones reales del conectoma) flotando al lado.
 */
export class FlyScene {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private clock = new THREE.Clock();
  private raf = 0;

  private fly = new THREE.Group();
  private head = new THREE.Group();
  private wings: THREE.Mesh[] = [];
  private frontLegs: THREE.Group[] = [];
  private book = new THREE.Group();
  private pageL!: THREE.Mesh;
  private pageR!: THREE.Mesh;
  private turning!: THREE.Mesh;
  private turnT = 1; // 0..1 animación de página
  private pageCanvasL = document.createElement("canvas");
  private pageCanvasR = document.createElement("canvas");
  private texL!: THREE.CanvasTexture;
  private texR!: THREE.CanvasTexture;
  private brain!: THREE.Points;
  private brainBase!: Float32Array;
  private brainGlow!: Float32Array;
  private brainGroup = new THREE.Group();

  /** velocidad de reproducción actual (afecta aleteo y lectura) */
  speed = 1;
  playing = true;
  private lastPage = -1;

  constructor(private host: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    host.appendChild(this.renderer.domElement);

    this.scene.background = new THREE.Color("#0b0b0b");
    this.scene.fog = new THREE.Fog("#0b0b0b", 9, 22);
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

    this.camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
    this.camera.position.set(3.2, 5.0, 8.6);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0.45, 1.5, 0);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.minDistance = 3.5;
    this.controls.maxDistance = 16;
    this.controls.maxPolarAngle = Math.PI * 0.49;
    this.controls.autoRotate = true;
    this.controls.autoRotateSpeed = 0.35;

    this.lights();
    this.desk();
    this.buildBook();
    this.buildFly();
    this.scene.add(this.brainGroup);
    this.brainGroup.position.set(1.0, 3.05, -2.4);
    this.brainGroup.scale.setScalar(1.45);

    this.resize();
    window.addEventListener("resize", this.resize);
    this.loop();
  }

  /* ---------------- luces y escritorio ---------------- */
  private lights() {
    const lamp = new THREE.SpotLight("#fff4e0", 260, 30, Math.PI / 5, 0.55, 1.6);
    lamp.position.set(2.5, 8, 3);
    lamp.target.position.set(0, 0, 0);
    lamp.castShadow = true;
    lamp.shadow.mapSize.set(2048, 2048);
    lamp.shadow.bias = -0.0004;
    lamp.shadow.radius = 6;
    this.scene.add(lamp, lamp.target);
    this.scene.add(new THREE.HemisphereLight("#ffffff", "#202020", 0.35));
    const rim = new THREE.DirectionalLight("#9fb6ff", 0.6);
    rim.position.set(-6, 4, -6);
    this.scene.add(rim);
  }

  private desk() {
    const g = new THREE.CircleGeometry(40, 64);
    const m = new THREE.MeshStandardMaterial({ color: "#151515", roughness: 0.85, metalness: 0 });
    const floor = new THREE.Mesh(g, m);
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    this.scene.add(floor);
  }

  /* ---------------- libro ---------------- */
  private buildBook() {
    const W = 2.1, H = 2.9;
    // tapas
    const coverM = new THREE.MeshStandardMaterial({ color: "#1d1d1d", roughness: 0.55, metalness: 0.1 });
    const cover = new THREE.Mesh(new THREE.BoxGeometry(W * 2 + 0.25, 0.06, H + 0.2), coverM);
    cover.position.y = 0.03;
    cover.castShadow = cover.receiveShadow = true;
    this.book.add(cover);
    // bloques de páginas
    const paperM = new THREE.MeshStandardMaterial({ color: "#e9e6dd", roughness: 0.95 });
    for (const s of [-1, 1]) {
      const block = new THREE.Mesh(new THREE.BoxGeometry(W, 0.22, H), paperM);
      block.position.set(s * (W / 2 + 0.03), 0.17, 0);
      block.castShadow = block.receiveShadow = true;
      this.book.add(block);
    }
    // páginas con texto (canvas)
    for (const c of [this.pageCanvasL, this.pageCanvasR]) { c.width = 768; c.height = 1060; }
    this.texL = new THREE.CanvasTexture(this.pageCanvasL);
    this.texR = new THREE.CanvasTexture(this.pageCanvasR);
    for (const t of [this.texL, this.texR]) { t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8; }
    const pageGeo = () => {
      const g = new THREE.PlaneGeometry(W, H, 24, 1);
      g.rotateX(-Math.PI / 2);
      // leve curvatura hacia el lomo
      const p = g.attributes.position;
      for (let i = 0; i < p.count; i++) {
        const x = p.getX(i);
        const u = (x + W / 2) / W; // 0..1
        p.setY(i, Math.sin(u * Math.PI) * 0.06);
      }
      g.computeVertexNormals();
      return g;
    };
    this.pageL = new THREE.Mesh(pageGeo(), new THREE.MeshStandardMaterial({ map: this.texL, roughness: 0.92 }));
    this.pageR = new THREE.Mesh(pageGeo(), new THREE.MeshStandardMaterial({ map: this.texR, roughness: 0.92 }));
    this.pageL.position.set(-(W / 2 + 0.03), 0.285, 0);
    this.pageR.position.set(W / 2 + 0.03, 0.285, 0);
    this.pageL.receiveShadow = this.pageR.receiveShadow = true;
    this.book.add(this.pageL, this.pageR);
    // página que se da vuelta (pivota en el lomo)
    const tg = new THREE.PlaneGeometry(W, H, 30, 1);
    tg.translate(W / 2, 0, 0);
    tg.rotateX(-Math.PI / 2);
    this.turning = new THREE.Mesh(tg, new THREE.MeshStandardMaterial({ color: "#efece4", roughness: 0.95, side: THREE.DoubleSide }));
    this.turning.position.set(0.03, 0.3, 0);
    this.turning.castShadow = true;
    this.turning.visible = false;
    (this.turning.geometry as THREE.BufferGeometry).userData.base = Float32Array.from(tg.attributes.position.array as Float32Array);
    this.book.add(this.turning);

    this.book.rotation.y = -0.08;
    this.scene.add(this.book);
  }

  /** dibuja el texto de las dos páginas; `cursor` = cuántos caracteres de la derecha ya leyó */
  setPages(left: string, right: string, cursor: number, title: string) {
    const draw = (c: HTMLCanvasElement, txt: string, upto: number, header: string) => {
      const x = c.getContext("2d")!;
      x.fillStyle = "#ece8de";
      x.fillRect(0, 0, c.width, c.height);
      x.fillStyle = "#9a958a";
      x.font = "italic 26px Georgia, serif";
      x.textAlign = "center";
      x.fillText(header, c.width / 2, 62);
      x.textAlign = "left";
      x.font = "29px Georgia, serif";
      const lines = wrap(x, txt, c.width - 120);
      let n = 0;
      lines.slice(0, 24).forEach((ln, i) => {
        const y = 128 + i * 38;
        let px = 60;
        for (const ch of ln) {
          const read = upto < 0 || n < upto;
          const isCursor = n === upto;
          if (isCursor) {
            x.fillStyle = "#0a0a0a";
            x.fillRect(px - 1, y - 27, x.measureText(ch).width + 2, 34);
            x.fillStyle = "#ece8de";
          } else x.fillStyle = read ? "#1a1a1a" : "#b9b4a8";
          x.fillText(ch, px, y);
          px += x.measureText(ch).width;
          n++;
        }
        n++; // espacio del salto de línea
      });
    };
    draw(this.pageCanvasL, left, -1, title);
    draw(this.pageCanvasR, right, cursor, title);
    this.texL.needsUpdate = true;
    this.texR.needsUpdate = true;
  }

  /** dispara el pase de página si cambió el número */
  setPageNumber(n: number) {
    if (this.lastPage !== -1 && n !== this.lastPage) this.turnT = 0;
    this.lastPage = n;
  }

  /* ---------------- mosca ---------------- */
  private buildFly() {
    const body = new THREE.MeshPhysicalMaterial({ color: "#1b1b1c", roughness: 0.38, metalness: 0.15, clearcoat: 1, clearcoatRoughness: 0.25 });
    const hairy = new THREE.MeshStandardMaterial({ color: "#2a2a2b", roughness: 0.75 });
    const eyeM = new THREE.MeshPhysicalMaterial({ color: "#7d1d16", roughness: 0.25, clearcoat: 1, clearcoatRoughness: 0.1, sheen: 1, sheenColor: new THREE.Color("#ff5a3d") });

    // tórax
    const thorax = new THREE.Mesh(new THREE.SphereGeometry(0.42, 48, 32), body);
    thorax.scale.set(1, 0.86, 1.12);
    thorax.castShadow = true;
    this.fly.add(thorax);
    // franjas del tórax (típicas de la mosca común)
    const stripeM = new THREE.MeshStandardMaterial({ color: "#5c5c5c", roughness: 0.6 });
    for (const sx of [-0.12, 0.12]) {
      const s = new THREE.Mesh(new THREE.SphereGeometry(0.43, 32, 16, 0, Math.PI * 2, 0, Math.PI / 2.3), stripeM);
      s.scale.set(0.12, 0.87, 1.1);
      s.position.x = sx;
      this.fly.add(s);
    }
    // abdomen con bandas
    const abdC = document.createElement("canvas");
    abdC.width = 64; abdC.height = 256;
    const ax = abdC.getContext("2d")!;
    for (let i = 0; i < 8; i++) { ax.fillStyle = i % 2 ? "#141412" : "#3b3934"; ax.fillRect(0, i * 32, 64, 32); }
    const abdTex = new THREE.CanvasTexture(abdC);
    abdTex.colorSpace = THREE.SRGBColorSpace;
    const abdomen = new THREE.Mesh(new THREE.SphereGeometry(0.4, 48, 32), new THREE.MeshPhysicalMaterial({ map: abdTex, roughness: 0.45, clearcoat: 0.5, clearcoatRoughness: 0.35 }));
    abdomen.scale.set(1.0, 0.82, 1.25);
    abdomen.position.set(0, -0.05, -0.78);
    abdomen.rotation.x = -0.25;
    abdomen.castShadow = true;
    this.fly.add(abdomen);

    // cabeza
    const headCore = new THREE.Mesh(new THREE.SphereGeometry(0.3, 40, 28), hairy);
    headCore.scale.set(1.25, 0.95, 0.8);
    this.head.add(headCore);
    for (const s of [-1, 1]) {
      const eye = new THREE.Mesh(new THREE.SphereGeometry(0.22, 40, 28), eyeM);
      eye.scale.set(0.9, 1.1, 0.95);
      eye.position.set(s * 0.25, 0.04, 0.03);
      eye.castShadow = true;
      this.head.add(eye);
      // antenas
      const ant = new THREE.Mesh(new THREE.CapsuleGeometry(0.025, 0.12, 4, 8), hairy);
      ant.position.set(s * 0.07, 0.08, 0.26);
      ant.rotation.set(0.9, 0, s * 0.3);
      this.head.add(ant);
    }
    // probóscide
    const prob = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.06, 0.25, 12), hairy);
    prob.position.set(0, -0.2, 0.18);
    prob.rotation.x = 0.6;
    this.head.add(prob);
    // anteojos de lectura 👓
    const glassM = new THREE.MeshStandardMaterial({ color: "#d8d8d8", metalness: 1, roughness: 0.2 });
    const lensM = new THREE.MeshPhysicalMaterial({ color: "#ffffff", transmission: 1, roughness: 0.02, thickness: 0.02, transparent: true, opacity: 0.35 });
    for (const s of [-1, 1]) {
      const rim = new THREE.Mesh(new THREE.TorusGeometry(0.13, 0.012, 10, 40), glassM);
      rim.position.set(s * 0.2, 0.03, 0.27);
      this.head.add(rim);
      const lens = new THREE.Mesh(new THREE.CircleGeometry(0.125, 32), lensM);
      lens.position.copy(rim.position);
      this.head.add(lens);
    }
    const bridge = new THREE.Mesh(new THREE.CylinderGeometry(0.01, 0.01, 0.14, 8), glassM);
    bridge.rotation.z = Math.PI / 2;
    bridge.position.set(0, 0.06, 0.28);
    this.head.add(bridge);

    this.head.position.set(0, 0.12, 0.55);
    this.head.rotation.x = 0.35;
    this.fly.add(this.head);

    // alas (iridiscentes, con venas)
    const wc = document.createElement("canvas");
    wc.width = 256; wc.height = 512;
    const wx = wc.getContext("2d")!;
    wx.strokeStyle = "rgba(40,40,40,0.9)";
    wx.lineWidth = 3;
    for (let i = 0; i < 6; i++) {
      wx.beginPath(); wx.moveTo(128, 0);
      wx.quadraticCurveTo(128 + (i - 2.5) * 40, 250, 128 + (i - 2.5) * 38, 512);
      wx.stroke();
    }
    const wingTex = new THREE.CanvasTexture(wc);
    wingTex.colorSpace = THREE.SRGBColorSpace;
    const wingShape = new THREE.Shape();
    wingShape.moveTo(0, 0);
    wingShape.bezierCurveTo(0.35, 0.05, 0.45, 0.8, 0.22, 1.25);
    wingShape.bezierCurveTo(0.05, 1.45, -0.2, 1.2, -0.12, 0.6);
    wingShape.bezierCurveTo(-0.08, 0.3, -0.04, 0.1, 0, 0);
    const wingGeo = new THREE.ShapeGeometry(wingShape, 24);
    // UVs simples
    const pos = wingGeo.attributes.position;
    const uv = new Float32Array(pos.count * 2);
    for (let i = 0; i < pos.count; i++) { uv[i * 2] = (pos.getX(i) + 0.3) / 0.8; uv[i * 2 + 1] = pos.getY(i) / 1.45; }
    wingGeo.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
    const wingM = new THREE.MeshPhysicalMaterial({
      color: "#ffffff", map: wingTex, transparent: true, opacity: 0.38, roughness: 0.15, side: THREE.DoubleSide,
      iridescence: 1, iridescenceIOR: 1.35, depthWrite: false,
    });
    for (const s of [-1, 1]) {
      const pivot = new THREE.Group();
      const w = new THREE.Mesh(wingGeo, wingM);
      w.rotation.x = -Math.PI / 2;
      w.scale.set(s, 1, 1);
      pivot.add(w);
      pivot.position.set(s * 0.18, 0.3, 0.05);
      pivot.userData.side = s;
      this.fly.add(pivot);
      this.wings.push(pivot as unknown as THREE.Mesh);
    }

    // patas: 3 por lado, 3 segmentos
    const legM = hairy;
    const seg = (len: number, r: number) => {
      const g = new THREE.Group();
      const m = new THREE.Mesh(new THREE.CapsuleGeometry(r, len, 4, 8), legM);
      m.position.y = -len / 2;
      m.castShadow = true;
      g.add(m);
      g.userData.len = len;
      return g;
    };
    const legDefs = [
      { z: 0.28, yaw: 0.5, front: true },
      { z: 0.0, yaw: 0.05, front: false },
      { z: -0.25, yaw: -0.5, front: false },
    ];
    for (const s of [-1, 1]) {
      for (const d of legDefs) {
        const hip = new THREE.Group();
        hip.position.set(s * 0.25, -0.22, d.z);
        const femur = seg(0.42, 0.04);
        const tibia = seg(0.45, 0.032);
        const tarsus = seg(0.28, 0.024);
        tibia.position.y = -0.46;
        tarsus.position.y = -0.48;
        femur.add(tibia);
        tibia.add(tarsus);
        hip.add(femur);
        if (d.front) {
          // patas delanteras: estiradas hacia el libro (para pasar páginas)
          hip.rotation.set(0.0, 0, 0);
          femur.rotation.set(-1.1, 0, s * 0.35);
          tibia.rotation.set(0.9, 0, 0);
          tarsus.rotation.set(0.6, 0, 0);
          hip.userData = { femur, tibia, side: s };
          this.frontLegs.push(hip);
        } else {
          femur.rotation.set(0, d.yaw * s, s * 1.0);
          tibia.rotation.set(0, 0, -s * 1.25);
          tarsus.rotation.set(0, 0, -s * 0.35);
        }
        this.fly.add(hip);
      }
    }

    this.fly.scale.setScalar(0.78);
    this.fly.position.set(-0.35, 0.95, 1.85);
    this.fly.rotation.set(0.1, Math.PI - 0.55, 0); // mirando al libro, de costado
    this.scene.add(this.fly);
  }

  /* ---------------- cerebro ---------------- */
  /** posiciones normalizadas [-1,1] (x,y,z intercaladas) */
  setBrain(positions: Float32Array, groups?: { inputs?: [number, number]; kenyon?: [number, number] }) {
    if (this.brain) { this.brainGroup.remove(this.brain); this.brain.geometry.dispose(); }
    const n = positions.length / 3;
    const g = new THREE.BufferGeometry();
    const p = new Float32Array(positions.length);
    for (let i = 0; i < n; i++) {
      p[i * 3] = positions[i * 3] * 1.9;
      p[i * 3 + 1] = positions[i * 3 + 1] * 1.0;
      p[i * 3 + 2] = positions[i * 3 + 2] * 0.8;
    }
    g.setAttribute("position", new THREE.BufferAttribute(p, 3));
    this.brainBase = new Float32Array(n).fill(0.18);
    this.brainGlow = new Float32Array(n).fill(0);
    g.setAttribute("glow", new THREE.BufferAttribute(new Float32Array(n), 1));
    const tint = new Float32Array(n);
    if (groups?.inputs) for (let i = groups.inputs[0]; i < groups.inputs[1]; i++) tint[i] = 1;
    if (groups?.kenyon) for (let i = groups.kenyon[0]; i < groups.kenyon[1]; i++) tint[i] = 2;
    g.setAttribute("tint", new THREE.BufferAttribute(tint, 1));
    const m = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: { uScale: { value: this.renderer.getPixelRatio() } },
      vertexShader: `
        attribute float glow; attribute float tint; varying float vG; varying float vT; uniform float uScale;
        void main(){ vG = glow; vT = tint; vec4 mv = modelViewMatrix * vec4(position,1.0);
          gl_PointSize = (3.0 + glow*8.0) * uScale * (7.0 / -mv.z); gl_Position = projectionMatrix * mv; }`,
      fragmentShader: `
        varying float vG; varying float vT;
        void main(){ vec2 c = gl_PointCoord - 0.5; float d = length(c); if(d>0.5) discard;
          float a = smoothstep(0.5, 0.0, d);
          vec3 base = vT > 1.5 ? vec3(0.95, 0.72, 0.35) : (vT > 0.5 ? vec3(0.45, 0.72, 0.95) : vec3(0.5));
          vec3 col = mix(base, vec3(1.0, 0.97, 0.9), vG);
          gl_FragColor = vec4(col, a * (0.32 + vG*0.9)); }`,
    });
    this.brain = new THREE.Points(g, m);
    // inclinado hacia la cámara para verlo de frente y no desde arriba
    this.brain.rotation.x = 0.45;
    this.brainGroup.add(this.brain);
  }

  /** índices de neuronas activas ahora (se encienden y se van apagando) */
  pulseNeurons(indices: number[], strength = 1) {
    if (!this.brainGlow) return;
    for (const i of indices) if (i < this.brainGlow.length) this.brainGlow[i] = Math.min(1, this.brainGlow[i] + strength);
  }

  /* ---------------- bucle ---------------- */
  private resize = () => {
    const w = this.host.clientWidth, h = this.host.clientHeight;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    // en pantallas angostas, alejar la cámara
    this.camera.fov = w < 700 ? 46 : 32;
    this.camera.updateProjectionMatrix();
  };

  private loop = () => {
    this.raf = requestAnimationFrame(this.loop);
    const dt = Math.min(this.clock.getDelta(), 0.05);
    const t = this.clock.elapsedTime;
    const sp = this.playing ? this.speed : 0;
    const energy = Math.min(1, Math.log10(1 + sp) / 3); // 0..1

    // respiración y cabeza que lee (barre la página izquierda → derecha)
    this.fly.position.y = 0.95 + Math.sin(t * 2) * 0.006;
    const sweep = this.playing ? Math.sin(t * (1.2 + energy * 5)) : 0;
    this.head.rotation.y = sweep * 0.25;
    this.head.rotation.x = 0.35 + Math.sin(t * 0.7) * 0.03;

    // alas: quietas plegadas; vibran con la velocidad
    const flap = energy > 0.55 ? Math.sin(t * 90) * 0.5 * (energy - 0.5) : Math.sin(t * 3) * 0.02;
    for (const w of this.wings) {
      const s = w.userData.side as number;
      w.rotation.set(0.12 + flap * 0.4, s * (0.42 + energy * 0.5) + flap * s, s * (0.06 + flap));
      w.rotation.order = "YXZ";
    }

    // patas delanteras: una "acompaña" la línea y la otra pasa la página
    for (const leg of this.frontLegs) {
      const { femur, side } = leg.userData as { femur: THREE.Group; side: number };
      femur.rotation.x = -1.1 + Math.sin(t * (2 + energy * 6) + side) * 0.08;
    }

    // pase de página
    if (this.turnT < 1) {
      this.turnT = Math.min(1, this.turnT + dt * (1.4 + energy * 3));
      this.turning.visible = true;
      const a = easeInOut(this.turnT) * Math.PI; // 0 → π
      const geo = this.turning.geometry as THREE.BufferGeometry;
      const base = geo.userData.base as Float32Array;
      const p = geo.attributes.position as THREE.BufferAttribute;
      const bend = Math.sin(this.turnT * Math.PI) * 0.55;
      for (let i = 0; i < p.count; i++) {
        const x0 = base[i * 3], z0 = base[i * 3 + 2];
        const u = x0 / 2.1;
        const ang = Math.PI - a + bend * u * (1 - u) * 2.2 * (this.turnT < 0.5 ? 1 : -1);
        p.setXYZ(i, Math.cos(ang) * x0 * -1, Math.sin(ang) * x0 * 0.9 + 0.001, z0);
      }
      p.needsUpdate = true;
      geo.computeVertexNormals();
      if (this.turnT >= 1) this.turning.visible = false;
    }

    // cerebro: rota suave, los puntos se apagan con el tiempo
    if (this.brain) {
      this.brainGroup.rotation.y += dt * 0.15;
      const glow = this.brain.geometry.attributes.glow as THREE.BufferAttribute;
      const decay = Math.exp(-dt * 2.2);
      for (let i = 0; i < this.brainGlow.length; i++) {
        this.brainGlow[i] *= decay;
        (glow.array as Float32Array)[i] = this.brainGlow[i] + this.brainBase[i] * 0;
      }
      glow.needsUpdate = true;
    }

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };

  /** detiene la rotación automática cuando el usuario toma la cámara */
  setAutoRotate(on: boolean) { this.controls.autoRotate = on; }

  dispose() {
    cancelAnimationFrame(this.raf);
    window.removeEventListener("resize", this.resize);
    this.controls.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}

function easeInOut(x: number) { return x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2; }

function wrap(ctx: CanvasRenderingContext2D, text: string, maxW: number) {
  const words = text.replace(/\s+/g, " ").split(" ");
  const lines: string[] = [];
  let cur = "";
  for (const w of words) {
    const t = cur ? cur + " " + w : w;
    if (ctx.measureText(t).width > maxW && cur) { lines.push(cur); cur = w; } else cur = t;
  }
  if (cur) lines.push(cur);
  return lines;
}
