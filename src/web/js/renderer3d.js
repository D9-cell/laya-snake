import * as THREE from "three";
import { Garden, MARGIN } from "./scene3d.js";
import { Snake3D } from "./snake3d.js";
import { Rabbits } from "./rabbit3d.js";
import { Centerline, buildCenterline } from "./snakeMotion.js";

// Camera presets: the 3D view shows the whole yard from high up; Realistic is lower, angled and closer.
const MODES = {
  "3d": { fov: 36, pitch: 55, yaw: 0, fit: 1, follow: 13 },
  cinematic: { fov: 30, pitch: 38, yaw: -16, fit: 0.84, follow: 10 },
};
const UP = new THREE.Vector3(0, 1, 0);
const _dir = new THREE.Vector3(), _pos = new THREE.Vector3(), _target = new THREE.Vector3(), _mat = new THREE.Matrix4(), _q = new THREE.Quaternion();

// Miniature-style depth of field: keeps a horizontal band around the snake sharp, blurs above and
// below, adds a light vignette, then tone-maps to the screen.
class TiltShift {
  constructor() {
    this.target = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4 });
    this.material = new THREE.ShaderMaterial({
      uniforms: { tDiffuse: { value: this.target.texture }, resolution: { value: new THREE.Vector2(1, 1) }, focusY: { value: 0.5 }, band: { value: 0.2 }, maxBlur: { value: 5 } },
      vertexShader: "varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }",
      fragmentShader: `
        uniform sampler2D tDiffuse; uniform vec2 resolution; uniform float focusY; uniform float band; uniform float maxBlur;
        varying vec2 vUv;
        const vec2 taps[12] = vec2[](vec2(-0.326,-0.406), vec2(-0.840,-0.074), vec2(-0.696,0.457), vec2(-0.203,0.621),
          vec2(0.962,-0.195), vec2(0.473,-0.480), vec2(0.519,0.767), vec2(0.185,-0.893), vec2(0.507,0.064),
          vec2(0.896,0.412), vec2(-0.322,-0.933), vec2(-0.792,-0.598));
        void main() {
          float blur = smoothstep(band, band + 0.32, abs(vUv.y - focusY)) * maxBlur;
          vec4 c = texture2D(tDiffuse, vUv);
          if (blur > 0.25) {
            vec4 acc = c;
            for (int i = 0; i < 12; i++) acc += texture2D(tDiffuse, vUv + taps[i] * blur / resolution);
            c = acc / 13.0;
          }
          vec2 q = vUv - 0.5;
          c.rgb *= 1.0 - dot(q, q) * 0.6;
          gl_FragColor = c;
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
      depthTest: false, depthWrite: false,
    });
    this.quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.material);
    this.quad.frustumCulled = false;
    this.scene = new THREE.Scene();
    this.scene.add(this.quad);
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  }

  setSize(w, h) {
    this.target.setSize(w, h);
    this.material.uniforms.resolution.value.set(w, h);
    this.material.uniforms.maxBlur.value = Math.max(2, h / 320);
  }

  render(renderer, scene, camera, focusY) {
    renderer.setRenderTarget(this.target);
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    this.material.uniforms.focusY.value = focusY;
    renderer.render(this.scene, this.camera);
  }

  dispose() { this.target.dispose(); this.material.dispose(); this.quad.geometry.dispose(); }
}

// The 3D game world. It only reads the server state it is given; it never touches the dashboard.
export class ThreeRenderer {
  constructor(canvas, prefs) {
    this.canvas = canvas;
    this.prefs = prefs;
    this.ready = false;
    this.mode = "3d";
    this.S = null;
    this.cl = new Centerline();
    this.look = new THREE.Vector3();
    this.camSnap = true;
    this.last = 0;
    this.debug = null;
  }

  // Creates the renderer, scene, lights and models (once).
  async init() {
    const r = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, powerPreference: "high-performance" });
    this.renderer = r;
    r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    r.outputColorSpace = THREE.SRGBColorSpace;
    r.toneMapping = THREE.ACESFilmicToneMapping;
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.PCFSoftShadowMap;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(MODES["3d"].fov, 1, 0.1, 400);
    this.garden = new Garden(this.scene, r);
    this.snake = new Snake3D(this.scene);
    this.rabbits = new Rabbits(this.scene);
    this.post = new TiltShift();
    await Promise.all([this.snake.load(), this.rabbits.load()]);
    this.ready = true;
    this.resize();
    if (this.S) { const s = this.S; this.S = null; this.update(s); }
  }

  // Switches between the "3d" overview and the "cinematic" (Realistic) presentation.
  setMode(mode) { this.mode = MODES[mode] ? mode : "3d"; }

  // Takes a new server snapshot: board size, rocks and food positions come straight from it.
  update(s) {
    const resized = !this.S || this.S.width !== s.width || this.S.height !== s.height;
    this.S = s;
    if (!this.ready) return;
    if (resized) {
      this.garden.build(s.width, s.height);
      this.rabbits.clear();
      this.camSnap = true;
      if (this.debug) this.debug.gridKey = "";
    }
    this.garden.setObstacles(s.obstacles);
    this.rabbits.setFoods(s.foods, s.width, s.height, performance.now());
  }

  // Draws one frame: interpolate the snake, animate, move the camera, render.
  render(now) {
    if (!this.ready || !this.S) return;
    const dt = this.last ? Math.min(100, now - this.last) : 16;
    this.last = now;
    const s = this.S, cinematic = this.mode === "cinematic";
    this.garden.setLighting(this.prefs.tod, cinematic);
    this.garden.setGrid(this.prefs.grid);
    this.garden.update(now);
    buildCenterline(s, now, this.cl);
    this.snake.update(this.cl, s, now, dt);
    this.rabbits.showZ = !cinematic;
    this.rabbits.update(now, dt);
    this.updateCamera(dt);
    if (this.debug) this.updateDebug(dt);
    if (cinematic) {
      _pos.set(this.cl.n ? this.cl.x[0] - s.width / 2 : 0, 0.2, this.cl.n ? this.cl.y[0] - s.height / 2 : 0).project(this.camera);
      this.post.render(this.renderer, this.scene, this.camera, Math.min(0.85, Math.max(0.15, _pos.y * 0.5 + 0.5)));
    } else {
      this.renderer.render(this.scene, this.camera);
    }
  }

  // Camera distance that fits the fenced yard in view at this pitch, yaw and aspect.
  fitDistance(W, H, M) {
    const pitch = THREE.MathUtils.degToRad(M.pitch), yaw = Math.abs(THREE.MathUtils.degToRad(M.yaw));
    const hw = W / 2 + MARGIN * 0.55, hd = H / 2 + MARGIN * 0.55;
    const tv = Math.tan(THREE.MathUtils.degToRad(this.camera.fov) / 2), th = tv * this.camera.aspect;
    const across = hw * Math.cos(yaw) + hd * Math.sin(yaw), deep = hd * Math.cos(yaw) + hw * Math.sin(yaw);
    const distV = (deep * Math.sin(pitch) + 0.9 * Math.cos(pitch)) / tv + deep * Math.cos(pitch);
    const distH = across / th + deep * Math.cos(pitch) * 0.6;
    return Math.max(distV, distH);
  }

  // Eases the camera toward the overview or toward the snake's head; never snaps except on a new board.
  updateCamera(dt) {
    if (this.debug && this.debug.orbit && this.debug.orbit.enabled) { this.debug.orbit.update(); return; }
    const s = this.S, M = MODES[this.mode], W = s.width, H = s.height, cam = this.camera;
    const a = this.camSnap ? 1 : 1 - Math.exp(-dt / 450);
    if (Math.abs(cam.fov - M.fov) > 0.01) { cam.fov += (M.fov - cam.fov) * a; cam.updateProjectionMatrix(); }
    const pitch = THREE.MathUtils.degToRad(M.pitch), yaw = THREE.MathUtils.degToRad(M.yaw);
    _dir.set(Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), Math.cos(yaw) * Math.cos(pitch));
    const fit = this.fitDistance(W, H, M);
    let dist = fit * M.fit;
    _target.set(0, 0, 0);
    if (this.prefs.follow && this.cl.n) {
      const reachX = Math.max(0, W / 2 - 1.5), reachZ = Math.max(0, H / 2 - 1);
      _target.set(THREE.MathUtils.clamp(this.cl.x[0] - W / 2, -reachX, reachX), 0, THREE.MathUtils.clamp(this.cl.y[0] - H / 2, -reachZ, reachZ));
      dist = Math.min(fit, M.follow);
    }
    if (s.game_over) dist *= 0.93;
    _pos.copy(_target).addScaledVector(_dir, dist);
    cam.position.lerp(_pos, a);
    this.look.lerp(_target, a);
    _mat.lookAt(cam.position, this.look, UP);
    _q.setFromRotationMatrix(_mat);
    if (this.camSnap) cam.quaternion.copy(_q); else cam.quaternion.slerp(_q, 1 - Math.exp(-dt / 160));
    this.camSnap = false;
  }

  // Keeps the drawing buffer, camera aspect and post-processing target in step with the canvas size.
  resize() {
    if (!this.renderer) return;
    const w = Math.max(1, this.canvas.clientWidth), h = Math.max(1, this.canvas.clientHeight);
    const pr = Math.min(window.devicePixelRatio || 1, 2);
    this.renderer.setPixelRatio(pr);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.post.setSize(Math.round(w * pr), Math.round(h * pr));
  }

  // Called when the 2D view takes over; the next frame eases back in from where the camera was.
  deactivate() { this.last = 0; }

  // Turns on the developer overlay: path, skeleton, grid, axes, orbit camera, FPS.
  async enableDebug(panel) {
    const d = this.debug = { panel, fps: 60, gridKey: "", flags: {}, orbit: null };
    const pathGeo = new THREE.BufferGeometry();
    pathGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(8192 * 3), 3));
    d.path = new THREE.Line(pathGeo, new THREE.LineBasicMaterial({ color: 0xffd23f, depthTest: false }));
    d.path.renderOrder = 10; d.path.frustumCulled = false;
    d.axes = new THREE.AxesHelper(3);
    d.grid = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x4ef296, transparent: true, opacity: 0.6 }));
    for (const o of [d.path, d.axes, d.grid]) { o.visible = false; this.scene.add(o); }
    const { OrbitControls } = await import("three/addons/controls/OrbitControls.js");
    d.orbit = new OrbitControls(this.camera, this.canvas);
    d.orbit.enabled = false;
    panel.querySelectorAll("input[data-dbg]").forEach((box) => {
      box.onchange = () => {
        d.flags[box.dataset.dbg] = box.checked;
        if (box.dataset.dbg === "orbit") { d.orbit.enabled = box.checked; d.orbit.target.copy(this.look); if (!box.checked) this.camSnap = false; }
      };
    });
  }

  // Refreshes the debug helpers and readouts.
  updateDebug(dt) {
    const d = this.debug, s = this.S, W = s.width, H = s.height;
    d.fps += (1000 / Math.max(1, dt) - d.fps) * 0.05;
    d.path.visible = !!d.flags.path;
    if (d.path.visible) {
      const a = d.path.geometry.attributes.position, n = Math.min(this.cl.n, 8192);
      for (let i = 0; i < n; i++) a.setXYZ(i, this.cl.x[i] - W / 2, 0.05, this.cl.y[i] - H / 2);
      a.needsUpdate = true;
      d.path.geometry.setDrawRange(0, n);
    }
    d.axes.visible = !!d.flags.axes;
    d.grid.visible = !!d.flags.grid;
    if (d.grid.visible && d.gridKey !== W + "x" + H) {
      d.gridKey = W + "x" + H;
      const pts = [];
      for (let i = 0; i <= W; i++) pts.push(i - W / 2, 0.02, -H / 2, i - W / 2, 0.02, H / 2);
      for (let j = 0; j <= H; j++) pts.push(-W / 2, 0.02, j - H / 2, W / 2, 0.02, j - H / 2);
      d.grid.geometry.dispose();
      d.grid.geometry = new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    }
    const root = this.snake.rig && this.snake.rig.root;
    if (d.flags.bones && root && (!d.bones || d.bonesRoot !== root)) {
      if (d.bones) { this.scene.remove(d.bones); d.bones.dispose(); }
      d.bones = new THREE.SkeletonHelper(root);
      d.bones.material.depthTest = false;
      d.bonesRoot = root;
      this.scene.add(d.bones);
    }
    if (d.bones) d.bones.visible = !!d.flags.bones;
    const q = (id, v) => { const el = d.panel.querySelector("#" + id); if (el && el.textContent !== v) el.textContent = v; };
    q("dbg-fps", d.fps.toFixed(0));
    const c = this.camera.position;
    q("dbg-cam", `${c.x.toFixed(1)}, ${c.y.toFixed(1)}, ${c.z.toFixed(1)}`);
    q("dbg-model", `${this.snake.describe()} · rabbit ${this.rabbits.kind}`);
  }

  // Releases every GPU resource this renderer created.
  destroy() {
    if (!this.renderer) return;
    this.snake.dispose();
    this.rabbits.dispose();
    this.garden.dispose();
    this.post.dispose();
    if (this.debug && this.debug.orbit) this.debug.orbit.dispose();
    this.renderer.dispose();
    this.ready = false;
  }
}
