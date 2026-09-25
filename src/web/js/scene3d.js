import * as THREE from "three";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import { stoneTextures, grassTextures, woodTextures } from "./textures3d.js";
import { hash } from "./util.js";

// Board cell (x, y) to world space: one cell is one unit, logical y runs along +Z, Y is up, and the
// board is centred on the origin.
export function gridToWorld(x, y, W, H, out = new THREE.Vector3()) {
  return out.set(x + 0.5 - W / 2, 0, y + 0.5 - H / 2);
}

export const MARGIN = 3.2;

// Time-of-day presets: sky/ground fill, sun, lanterns, fog and exposure.
const LIGHTING = {
  day: {
    sky: 0xe6eefc, ground: 0x55603c, hemi: 0.95, sun: 0xfff2de, sunI: 2.7, sunDir: [-0.55, 1, 0.5],
    lantern: 0, glass: 0.25, bg: 0x93a79a, exposure: 0.95, env: 0.6,
    envTop: 0x8fb4e0, envHorizon: 0xf0ead8, envBottom: 0x3c4a2c,
  },
  golden: {
    sky: 0xffd6a8, ground: 0x3e3524, hemi: 0.7, sun: 0xffb56e, sunI: 2.6, sunDir: [-1, 0.52, 0.35],
    lantern: 2.2, glass: 2.2, bg: 0x6e5a48, exposure: 1.0, env: 0.5,
    envTop: 0x6a78a8, envHorizon: 0xffb877, envBottom: 0x2e2618,
  },
  night: {
    sky: 0x6479a8, ground: 0x1c2430, hemi: 0.75, sun: 0xb4c6ff, sunI: 1.15, sunDir: [0.45, 1, -0.35],
    lantern: 7, glass: 5, bg: 0x0d131c, exposure: 1.3, env: 0.38,
    envTop: 0x0c1428, envHorizon: 0x26324e, envBottom: 0x07090d,
  },
};

// Tileable 3D-ish noise from sines, used to roughen rocks and bushes (runs once per geometry).
const noise3 = (x, y, z) => (Math.sin(x * 3.1 + Math.sin(y * 2.3)) + Math.sin(y * 4.7 + Math.sin(z * 3.9)) + Math.sin(z * 5.3 + Math.sin(x * 4.1))) / 3;

// A rounded, weathered boulder about one cell across, with moss on the upward faces.
function rockGeometry(seed) {
  const geo = new THREE.IcosahedronGeometry(1, 3);
  const p = geo.attributes.position, colors = new Float32Array(p.count * 3), v = new THREE.Vector3();
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i);
    const n = noise3(v.x * 1.3 + seed, v.y * 1.3 - seed, v.z * 1.3 + seed * 2);
    const fine = noise3(v.x * 5 + seed, v.y * 5, v.z * 5 - seed) * 0.05;
    v.multiplyScalar(1 + n * 0.16 + fine);
    v.set(v.x * 0.43, v.y * 0.4, v.z * 0.43);
    if (v.y < 0) v.y *= 0.35;
    v.y += 0.12;
    p.setXYZ(i, v.x, v.y, v.z);
  }
  geo.computeVertexNormals();
  const nrm = geo.attributes.normal;
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i);
    const n = noise3(v.x * 6 + seed, v.y * 6, v.z * 6);
    const base = 0.13 + n * 0.035 + (hash(i, seed, 3) - 0.5) * 0.02;
    let c = [base * 1.06, base, base * 0.86];
    const moss = Math.max(0, nrm.getY(i) - 0.5) * 2 * (n > -0.1 ? 1 : 0.25);
    c = c.map((k, j) => k + ([0.045, 0.09, 0.022][j] - k) * Math.min(0.85, moss));
    colors.set(c, i * 3);
  }
  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  return geo;
}

// A lumpy leafy mound for bushes.
function bushGeometry() {
  const geo = new THREE.IcosahedronGeometry(1, 2), p = geo.attributes.position, v = new THREE.Vector3();
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i);
    v.multiplyScalar(1 + noise3(v.x * 2.4, v.y * 2.4, v.z * 2.4) * 0.22 + noise3(v.x * 9, v.y * 9, v.z * 9) * 0.06);
    if (v.y < 0) v.y *= 0.5;
    p.setXYZ(i, v.x, v.y, v.z);
  }
  geo.computeVertexNormals();
  return geo;
}

// A tuft of thin grass blades.
function tuftGeometry() {
  const blades = [];
  for (let i = 0; i < 7; i++) {
    const b = new THREE.ConeGeometry(0.012, 0.16 + hash(i, 1) * 0.12, 3).translate(0, 0.08, 0);
    b.rotateZ((hash(i, 2) - 0.5) * 0.7).rotateX((hash(i, 3) - 0.5) * 0.7).translate((hash(i, 4) - 0.5) * 0.08, 0, (hash(i, 5) - 0.5) * 0.08);
    blades.push(b);
  }
  return mergeGeometries(blades);
}

// Gradient sky with a sun disc, used only to build the reflection/ambient environment map.
function skyScene(preset) {
  const scene = new THREE.Scene();
  const mat = new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false,
    uniforms: { top: { value: new THREE.Color(preset.envTop) }, horizon: { value: new THREE.Color(preset.envHorizon) }, bottom: { value: new THREE.Color(preset.envBottom) } },
    vertexShader: "varying vec3 vDir; void main() { vDir = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
    fragmentShader: "uniform vec3 top; uniform vec3 horizon; uniform vec3 bottom; varying vec3 vDir; void main() { float h = vDir.y; vec3 c = h > 0.0 ? mix(horizon, top, pow(h, 0.6)) : mix(horizon, bottom, pow(-h, 0.4)); gl_FragColor = vec4(c, 1.0); }",
  });
  scene.add(new THREE.Mesh(new THREE.SphereGeometry(50, 32, 16), mat));
  const sun = new THREE.Mesh(new THREE.SphereGeometry(3, 16, 8), new THREE.MeshBasicMaterial({ color: new THREE.Color(preset.sun).multiplyScalar(preset.sunI * 4) }));
  sun.position.set(...preset.sunDir).normalize().multiplyScalar(40);
  scene.add(sun);
  return scene;
}

// The miniature garden around and under the board, plus its lights.
export class Garden {
  constructor(scene, renderer) {
    this.scene = scene;
    this.renderer = renderer;
    this.pmrem = new THREE.PMREMGenerator(renderer);
    this.envMaps = {};
    this.board = null;
    this.W = 0; this.H = 0;
    this.grid = true;
    this.obstacleKey = "";

    const stone = stoneTextures(), grass = grassTextures(), wood = woodTextures();
    grass.map.repeat.set(70, 70); grass.normalMap.repeat.set(70, 70);
    this.mat = {
      grass: new THREE.MeshStandardMaterial({ map: grass.map, normalMap: grass.normalMap, roughness: 0.95 }),
      tile: new THREE.MeshStandardMaterial({ map: stone.map, normalMap: stone.normalMap, normalScale: new THREE.Vector2(0.8, 0.8), roughnessMap: stone.roughnessMap, roughness: 1 }),
      grout: new THREE.MeshStandardMaterial({ color: 0x3a3a26, roughness: 1 }),
      moss: new THREE.MeshStandardMaterial({ color: 0x3d5424, roughness: 1 }),
      rock: new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.92, normalMap: stone.normalMap, normalScale: new THREE.Vector2(0.6, 0.6) }),
      wood: new THREE.MeshStandardMaterial({ map: wood.map, normalMap: wood.normalMap, roughness: 0.85 }),
      darkWood: new THREE.MeshStandardMaterial({ color: 0x2e2117, roughness: 0.8 }),
      glass: new THREE.MeshStandardMaterial({ color: 0xffe2a8, emissive: 0xffb55c, emissiveIntensity: 1, roughness: 0.3 }),
      leaf: new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.85 }),
      blade: new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.9 }),
      bloom: new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.7 }),
      water: new THREE.MeshPhysicalMaterial({ color: 0x173530, roughness: 0.05, metalness: 0, clearcoat: 1, clearcoatRoughness: 0.03, envMapIntensity: 1.4 }),
      pad: new THREE.MeshStandardMaterial({ color: 0x3f6e2a, roughness: 0.6, side: THREE.DoubleSide }),
      lotus: new THREE.MeshStandardMaterial({ color: 0xf2b3c8, roughness: 0.6 }),
    };
    const tileGeos = [0, 1, 2, 3].map((k) => {
      const g = new THREE.BoxGeometry(1, 0.08, 1).translate(0, -0.04, 0), uv = g.attributes.uv;
      for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) * 0.5 + (k & 1) * 0.5, uv.getY(i) * 0.5 + (k >> 1) * 0.5);
      return g;
    });
    this.geo = {
      tiles: tileGeos, rocks: [rockGeometry(1.3), rockGeometry(4.1), rockGeometry(7.7)], bush: bushGeometry(), tuft: tuftGeometry(),
      blob: new THREE.SphereGeometry(1, 10, 6), post: new THREE.BoxGeometry(0.1, 0.72, 0.1).translate(0, 0.36, 0),
      rail: new THREE.BoxGeometry(1, 0.05, 0.035), box: new THREE.BoxGeometry(1, 1, 1), roof: new THREE.ConeGeometry(0.24, 0.16, 4).rotateY(Math.PI / 4),
      pad: new THREE.CircleGeometry(1, 20, 0.35, Math.PI * 2 - 0.35).rotateX(-Math.PI / 2), water: new THREE.CircleGeometry(1, 64).rotateX(-Math.PI / 2),
    };

    const ground = new THREE.Mesh(new THREE.PlaneGeometry(260, 260).rotateX(-Math.PI / 2), this.mat.grass);
    ground.position.y = -0.045;
    ground.receiveShadow = true;
    scene.add(ground);
    this.ground = ground;

    this.hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 1);
    this.sun = new THREE.DirectionalLight(0xffffff, 2);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    this.sun.shadow.bias = -0.0004;
    this.sun.shadow.normalBias = 0.025;
    this.sun.shadow.radius = 3;
    scene.add(this.hemi, this.sun, this.sun.target);
    scene.fog = new THREE.Fog(0x000000, 40, 120);
    this.lanterns = [];
    this.tod = null;
  }

  // Rebuilds everything that depends on the board size: paving, fence, lanterns, plants and pond.
  build(W, H) {
    if (this.board) this.disposeBoard();
    this.W = W; this.H = H;
    this.board = new THREE.Group();
    this.board.name = "environment";
    this.scene.add(this.board);
    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(), p = new THREE.Vector3(), c = new THREE.Color();
    const inst = (geo, mat, count, shadow = true) => {
      const im = new THREE.InstancedMesh(geo, mat, Math.max(1, count));
      im.castShadow = shadow; im.receiveShadow = true;
      im.count = 0;
      this.board.add(im);
      return im;
    };
    const put = (im, x, y, z, sx, sy, sz, ry = 0, color = null) => {
      q.setFromAxisAngle(THREE.Object3D.DEFAULT_UP, ry);
      m.compose(p.set(x, y, z), q, s.set(sx, sy, sz));
      im.setMatrixAt(im.count, m);
      if (color !== null) im.setColorAt(im.count, c.set(color));
      im.count++;
    };

    const grout = new THREE.Mesh(this.geo.box, this.mat.grout);
    grout.scale.set(W + 0.3, 0.06, H + 0.3); grout.position.y = -0.055;
    grout.receiveShadow = true;
    this.board.add(grout);

    this.tiles = this.geo.tiles.map((g) => inst(g, this.mat.tile, W * H, false));
    this.layTiles();

    const moss = inst(this.geo.blob, this.mat.moss, (W + 1) * (H + 1), false);
    for (let j = 0; j <= H; j++) for (let i = 0; i <= W; i++) {
      if (hash(i, j, 8) > 0.18) continue;
      const r = 0.04 + hash(i, j, 9) * 0.05;
      put(moss, i - W / 2, -0.01, j - H / 2, r, 0.03, r * (0.7 + hash(i, j, 10) * 0.6), hash(i, j, 11) * 3, null);
    }

    const cap = Math.ceil(W * H * 0.08) + 8;
    this.rocks = this.geo.rocks.map((g) => inst(g, this.mat.rock, cap));
    this.obstacleKey = "";

    const posts = inst(this.geo.post, this.mat.wood, 2 * (W + H) + 4);
    const rails = inst(this.geo.rail, this.mat.wood, 4 * (W + H) + 8);
    const ex = W / 2 + 0.15, ez = H / 2 + 0.15;
    for (let i = 0; i <= W; i++) for (const z of [-ez, ez]) {
      const x = i - W / 2;
      if (i > 0 && i < W) put(posts, x, 0, z, 1, 0.95 + hash(i, z > 0, 1) * 0.1, 1, (hash(i, z > 0, 2) - 0.5) * 0.08);
      if (i < W) for (const y of [0.3, 0.58]) put(rails, x + 0.5, y, z, 1, 1, 1, 0);
    }
    for (let j = 0; j <= H; j++) for (const x of [-ex, ex]) {
      const z = j - H / 2;
      if (j > 0 && j < H) put(posts, x, 0, z, 1, 0.95 + hash(j, x > 0, 3) * 0.1, 1, (hash(j, x > 0, 4) - 0.5) * 0.08);
      if (j < H) for (const y of [0.3, 0.58]) put(rails, x, y, z + 0.5, 1, 1, 1, Math.PI / 2);
    }

    this.lanterns = [];
    for (const [x, z] of [[-ex, -ez], [ex, -ez], [-ex, ez], [ex, ez]]) {
      const g = new THREE.Group();
      g.position.set(x, 0, z);
      const part = (mat, sx, sy, sz, y) => { const o = new THREE.Mesh(this.geo.box, mat); o.scale.set(sx, sy, sz); o.position.y = y; o.castShadow = true; g.add(o); return o; };
      part(this.mat.wood, 0.16, 1.15, 0.16, 0.575);
      part(this.mat.darkWood, 0.34, 0.05, 0.34, 1.17);
      part(this.mat.glass, 0.2, 0.26, 0.2, 1.33).castShadow = false;
      for (const [dx, dz] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) part(this.mat.darkWood, 0.03, 0.28, 0.03, 1.33).position.set(dx * 0.1, 1.33, dz * 0.1);
      part(this.mat.darkWood, 0.3, 0.04, 0.3, 1.48);
      const roof = new THREE.Mesh(this.geo.roof, this.mat.darkWood);
      roof.position.y = 1.58; roof.castShadow = true;
      g.add(roof);
      const light = new THREE.PointLight(0xffb45e, 0, 9, 1.6);
      light.position.y = 1.33;
      g.add(light);
      this.board.add(g);
      this.lanterns.push(light);
    }

    const pond = { x: Math.max(-W / 2 + 2, W / 2 - Math.min(6, W * 0.3)), z: ez + MARGIN * 0.52, rx: Math.min(2.2, W * 0.12 + 0.8), rz: MARGIN * 0.3 };
    const inPond = (x, z, pad = 1.25) => ((x - pond.x) / (pond.rx * pad)) ** 2 + ((z - pond.z) / (pond.rz * pad)) ** 2 < 1;
    const water = new THREE.Mesh(this.geo.water, this.mat.water);
    water.scale.set(pond.rx, 1, pond.rz); water.position.set(pond.x, -0.02, pond.z);
    water.receiveShadow = true;
    this.board.add(water);
    const bed = new THREE.Mesh(this.geo.water, this.mat.grout);
    bed.scale.set(pond.rx * 1.08, 1, pond.rz * 1.12); bed.position.set(pond.x, -0.035, pond.z);
    this.board.add(bed);
    const pads = inst(this.geo.pad, this.mat.pad, 8, false), lotus = inst(this.geo.blob, this.mat.lotus, 3);
    for (let k = 0; k < 6; k++) {
      const a = hash(k, 5, 1) * Math.PI * 2, rr = 0.25 + hash(k, 5, 2) * 0.55;
      const x = pond.x + Math.cos(a) * pond.rx * rr, z = pond.z + Math.sin(a) * pond.rz * rr, r = 0.14 + hash(k, 5, 3) * 0.1;
      put(pads, x, -0.012, z, r, 1, r, a, null);
      if (k < 3) put(lotus, x + 0.03, 0.03, z, 0.06, 0.05, 0.06, 0, null);
    }

    const perim = 2 * (W + H) + 8 * MARGIN;
    const ring = (k, salt, minOut, maxOut) => {
      const t = hash(k, salt, 1) * perim, out = minOut + hash(k, salt, 2) * (maxOut - minOut);
      const hw = W / 2 + out, hh = H / 2 + out, side = 2 * (hw + hh), u = (t / perim) * side * 2;
      if (u < 2 * hw) return [u - hw, -hh];
      if (u < 2 * hw + 2 * hh) return [hw, u - 2 * hw - hh];
      if (u < 4 * hw + 2 * hh) return [hw - (u - 2 * hw - 2 * hh), hh];
      return [-hw, hh - (u - 4 * hw - 2 * hh)];
    };
    const bushes = inst(this.geo.bush, this.mat.leaf, Math.ceil(perim * 0.9));
    for (let k = 0; k < perim * 0.9; k++) {
      const [x, z] = ring(k, 31, 0.8, MARGIN - 0.2);
      if (inPond(x, z)) continue;
      const r = 0.22 + hash(k, 31, 3) * 0.3;
      put(bushes, x, 0, z, r, r * (0.7 + hash(k, 31, 4) * 0.3), r, hash(k, 31, 5) * 6, [0x2c4a1f, 0x365a24, 0x46692a, 0x2f5227][Math.floor(hash(k, 31, 6) * 4)]);
    }
    const tufts = inst(this.geo.tuft, this.mat.blade, Math.ceil(perim * 7), false);
    for (let k = 0; k < perim * 7; k++) {
      const [x, z] = ring(k, 37, 0.35, MARGIN + 3);
      if (inPond(x, z, 1.05)) continue;
      const r = 0.8 + hash(k, 37, 3) * 0.8;
      put(tufts, x, -0.03, z, r, r, r, hash(k, 37, 4) * 6, [0x4c7a2c, 0x5f8a34, 0x3d6624, 0x6f8f3a][Math.floor(hash(k, 37, 5) * 4)]);
    }
    const flowers = inst(this.geo.blob, this.mat.bloom, Math.ceil(perim * 1.2), false);
    for (let k = 0; k < perim * 1.2; k++) {
      const [x, z] = ring(k, 41, 0.45, MARGIN);
      if (inPond(x, z, 1.1)) continue;
      put(flowers, x, 0.1 + hash(k, 41, 3) * 0.08, z, 0.035, 0.028, 0.035, 0, [0xf4b6cf, 0xfff3dc, 0xffd66b, 0xc9b6ff][Math.floor(hash(k, 41, 4) * 4)]);
    }
    const pebbles = inst(this.geo.rocks[0], this.mat.rock, 40);
    for (let k = 0; k < 40; k++) {
      const a = k / 40 * Math.PI * 2 + hash(k, 43, 1) * 0.1, r = 0.22 + hash(k, 43, 2) * 0.14;
      put(pebbles, pond.x + Math.cos(a) * pond.rx * 1.06, -0.05, pond.z + Math.sin(a) * pond.rz * 1.08, r, r * 0.7, r, hash(k, 43, 3) * 6, null);
    }
    for (const im of this.board.children) if (im.isInstancedMesh) { im.instanceMatrix.needsUpdate = true; if (im.instanceColor) im.instanceColor.needsUpdate = true; }

    const R = Math.max(W, H) / 2 + MARGIN + 1;
    const sc = this.sun.shadow.camera;
    sc.left = sc.bottom = -R; sc.right = sc.top = R; sc.near = 1; sc.far = 120;
    sc.updateProjectionMatrix();
    this.scene.fog.near = R * 3.2; this.scene.fog.far = R * 9;
    this.tod = null;
  }

  // Lays the paving; with the grid off the joints close up so the floor reads as one surface.
  layTiles() {
    const W = this.W, H = this.H, size = this.grid ? 0.95 : 0.985;
    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), p = new THREE.Vector3(), s = new THREE.Vector3(), c = new THREE.Color();
    this.tiles.forEach((im) => { im.count = 0; });
    for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
      const im = this.tiles[Math.floor(hash(i, j, 12) * 4)];
      q.setFromAxisAngle(THREE.Object3D.DEFAULT_UP, Math.floor(hash(i, j, 13) * 4) * Math.PI / 2 + (hash(i, j, 14) - 0.5) * 0.02);
      m.compose(p.set(i + 0.5 - W / 2, (hash(i, j, 15) - 0.5) * 0.008, j + 0.5 - H / 2), q, s.set(size, 1, size));
      im.setMatrixAt(im.count, m);
      const l = 0.78 + hash(i, j, 16) * 0.22, g = hash(i, j, 17) < 0.12 ? 0.06 : 0;
      im.setColorAt(im.count, c.setRGB(l * (1 - g), l, l * (0.94 - g)));
      im.count++;
    }
    this.tiles.forEach((im) => { im.instanceMatrix.needsUpdate = true; if (im.instanceColor) im.instanceColor.needsUpdate = true; });
  }

  // Shows or hides the paving joints.
  setGrid(on) {
    if (on === this.grid) return;
    this.grid = on;
    if (this.board) this.layTiles();
  }

  // Places one boulder on every obstacle cell (only when the layout changes).
  setObstacles(list) {
    const key = list.join(";");
    if (key === this.obstacleKey || !this.board) return;
    this.obstacleKey = key;
    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), p = new THREE.Vector3(), s = new THREE.Vector3();
    this.rocks.forEach((im) => { im.count = 0; });
    for (const [x, y] of list) {
      const im = this.rocks[Math.floor(hash(x, y, 21) * 3)];
      if (im.count >= im.instanceMatrix.count) continue;
      q.setFromAxisAngle(THREE.Object3D.DEFAULT_UP, hash(x, y, 22) * Math.PI * 2);
      const w = 0.94 + hash(x, y, 23) * 0.08;
      m.compose(p.set(x + 0.5 - this.W / 2, -0.02, y + 0.5 - this.H / 2), q, s.set(w, 0.85 + hash(x, y, 24) * 0.4, w));
      im.setMatrixAt(im.count++, m);
    }
    this.rocks.forEach((im) => { im.instanceMatrix.needsUpdate = true; im.computeBoundingSphere(); });
  }

  // Applies a time-of-day preset; the realistic mode adds contrast and firmer shadows.
  setLighting(tod, cinematic) {
    const key = tod + (cinematic ? "+" : "");
    if (key === this.tod) return;
    this.tod = key;
    const L = LIGHTING[tod] || LIGHTING.day;
    this.hemi.color.set(L.sky); this.hemi.groundColor.set(L.ground);
    this.hemi.intensity = L.hemi * (cinematic ? 0.75 : 1);
    this.sun.color.set(L.sun);
    this.sun.intensity = L.sunI * (cinematic ? 1.2 : 1);
    this.sun.shadow.radius = cinematic ? 2 : 4;
    const R = Math.max(this.W, this.H) / 2 + MARGIN + 1;
    this.sun.position.set(...L.sunDir).normalize().multiplyScalar(R * 2.2);
    this.lanternLevel = L.lantern;
    this.mat.glass.emissiveIntensity = L.glass;
    this.scene.background = new THREE.Color(L.bg);
    this.scene.fog.color.set(L.bg);
    this.renderer.toneMappingExposure = L.exposure;
    if (!this.envMaps[tod]) {
      const sky = skyScene(L);
      this.envMaps[tod] = this.pmrem.fromScene(sky, 0.04).texture;
      sky.traverse((o) => { if (o.isMesh) { o.geometry.dispose(); o.material.dispose(); } });
    }
    this.scene.environment = this.envMaps[tod];
    this.scene.environmentIntensity = L.env;
  }

  // Lantern flicker.
  update(now) {
    const f = 1 + 0.06 * Math.sin(now / 83) + 0.04 * Math.sin(now / 37);
    for (const l of this.lanterns) l.intensity = (this.lanternLevel || 0) * f;
  }

  // Releases the per-board instance buffers and meshes.
  disposeBoard() {
    this.board.traverse((o) => { if (o.isInstancedMesh) o.dispose(); });
    this.board.removeFromParent();
    this.board = null;
  }

  dispose() {
    if (this.board) this.disposeBoard();
    Object.values(this.envMaps).forEach((t) => t.dispose());
    this.pmrem.dispose();
    [...this.geo.tiles, ...this.geo.rocks, this.geo.bush, this.geo.tuft, this.geo.blob, this.geo.post, this.geo.rail, this.geo.box, this.geo.roof, this.geo.pad, this.geo.water].forEach((g) => g.dispose());
    Object.values(this.mat).forEach((mt) => mt.dispose());
    this.ground.geometry.dispose();
  }
}
