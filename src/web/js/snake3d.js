import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { snakeSkinTextures } from "./textures3d.js";

export const SNAKE_URL = "/assets/snake.glb";

const RADIUS = 0.27;
const HEAD_RADIUS = 0.3;
// Snout-to-neck length (cells). The head never stretches; only the body behind it does.
const NECK = 0.62;
// Where the model's ends sit relative to the first and last cell centres.
const TIP_AHEAD = 0.42;
const TAIL_BEHIND = 0.38;
const UP = new THREE.Vector3(0, 1, 0);

const _x = new THREE.Vector3(), _z = new THREE.Vector3(), _f = new THREE.Vector3(), _p = new THREE.Vector3();
const _m = new THREE.Matrix4(), _q = new THREE.Quaternion(), _s = new THREE.Vector3();
const _a = { x: 0, y: 0 }, _b = { x: 0, y: 0 }, _c = { x: 0, y: 0 }, _e = { x: 0, y: 0 };
const _inv = new THREE.Matrix4();
const smooth = (e0, e1, x) => { const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0))); return t * t * (3 - 2 * t); };

// Orientation whose -Z axis points along the horizontal `forward` direction, with +Y up.
function bodyFrame(forward, out) {
  _z.copy(forward).negate();
  _x.crossVectors(UP, _z).normalize();
  _m.makeBasis(_x, UP, _z);
  return out.setFromRotationMatrix(_m);
}

// Half-width and half-height of the procedural body `s` cells behind the snout tip.
function profile(s, L) {
  if (s < NECK) {
    const snout = s < 0.34 ? Math.sqrt(Math.max(0, 1 - Math.pow(1 - s / 0.34, 2))) : 1;
    const w = HEAD_RADIUS * snout * (1 - smooth(0.4, NECK, s) * 0.2);
    return [w * 1.12, w * 0.7];
  }
  const body = Math.max(0.01, L - NECK), tail = Math.min(body * 0.45, Math.max(1.2, body * 0.2 + 0.8));
  const grow = smooth(NECK, NECK + Math.min(1.2, body * 0.15), s), taper = smooth(L - tail, L, s);
  const r = RADIUS * (0.86 + 0.14 * grow) * (1 - Math.pow(taper, 1.1) * 0.93);
  return [r * 1.08, r * 0.86];
}

let parts = null;
// Geometry and materials shared by every procedural snake build (created once).
function sharedParts() {
  if (parts) return parts;
  const skin = snakeSkinTextures();
  skin.normalMap.repeat.set(1, 1);
  parts = {
    skin: new THREE.MeshPhysicalMaterial({
      map: skin.map, normalMap: skin.normalMap, normalScale: new THREE.Vector2(0.5, 0.5), roughnessMap: skin.roughnessMap,
      roughness: 1, metalness: 0, clearcoat: 0.28, clearcoatRoughness: 0.42, specularIntensity: 0.6,
    }),
    eye: new THREE.MeshPhysicalMaterial({ color: 0x5c4414, roughness: 0.12, clearcoat: 1, clearcoatRoughness: 0.04 }),
    pupil: new THREE.MeshPhysicalMaterial({ color: 0x050505, roughness: 0.08, clearcoat: 1, clearcoatRoughness: 0.04 }),
    dark: new THREE.MeshStandardMaterial({ color: 0x17120c, roughness: 0.9 }),
    tongue: new THREE.MeshPhysicalMaterial({ color: 0x9c1f35, roughness: 0.35, clearcoat: 0.5, clearcoatRoughness: 0.3 }),
    eyeGeo: new THREE.SphereGeometry(1, 20, 14),
    shaftGeo: new THREE.CylinderGeometry(0.008, 0.011, 0.16, 6).rotateX(Math.PI / 2).translate(0, 0, -0.08),
    forkGeo: new THREE.CylinderGeometry(0.004, 0.007, 0.07, 5).rotateX(Math.PI / 2).translate(0, 0, -0.035),
  };
  return parts;
}

// A forked tongue pointing along -Z from its origin; animate `scale.z` to flick it.
function makeTongue() {
  const p = sharedParts(), g = new THREE.Group();
  g.name = "Tongue";
  g.add(new THREE.Mesh(p.shaftGeo, p.tongue));
  for (const s of [1, -1]) {
    const fork = new THREE.Mesh(p.forkGeo, p.tongue);
    fork.position.z = -0.155; fork.rotation.y = s * 0.38;
    g.add(fork);
  }
  g.scale.z = 0.01; g.visible = false;
  return g;
}

// Builds a straight, rigged snake of rest length `L` cells lying along +Z with the snout at z = 0:
// SnakeRoot > Body (SkinnedMesh) > Bone_01 > Bone_02 > ... with eyes, nostrils, mouth and tongue
// parented to the head bone.
function buildProceduralSnake(L) {
  const p = sharedParts();
  const boneS = [0.12, 0.36, NECK];
  for (let s = NECK + 0.3; s < L - 0.15; s += 0.3) boneS.push(s);
  boneS.push(L - 0.08);

  const SEG = 22, rings = [];
  for (let s = 0; s < L; s += s < NECK + 0.2 ? 0.035 : s > L - 1 ? 0.05 : 0.1) rings.push(s);
  rings.push(L);
  const vCount = rings.length * (SEG + 1);
  const pos = new Float32Array(vCount * 3), uv = new Float32Array(vCount * 2);
  const skinIndex = new Uint16Array(vCount * 4), skinWeight = new Float32Array(vCount * 4);
  const circ = 2 * Math.PI * RADIUS * 0.97;
  let b = 0;
  rings.forEach((s, r) => {
    const [rx, ry] = profile(s, L), cy = ry * 0.8;
    while (b < boneS.length - 2 && boneS[b + 1] <= s) b++;
    const w = s <= boneS[0] ? 0 : Math.min(1, Math.max(0, (s - boneS[b]) / (boneS[b + 1] - boneS[b])));
    for (let j = 0; j <= SEG; j++) {
      const th = j / SEG * Math.PI * 2, c = Math.cos(th), i = r * (SEG + 1) + j;
      const y = c > 0.6 ? cy - ry * (0.6 + (c - 0.6) * 0.45) : cy - ry * c;
      pos[i * 3] = rx * Math.sin(th); pos[i * 3 + 1] = y; pos[i * 3 + 2] = s;
      uv[i * 2] = j / SEG; uv[i * 2 + 1] = s / circ;
      skinIndex[i * 4] = b; skinIndex[i * 4 + 1] = b + 1;
      skinWeight[i * 4] = 1 - w; skinWeight[i * 4 + 1] = w;
    }
  });
  const index = [];
  for (let r = 0; r < rings.length - 1; r++) for (let j = 0; j < SEG; j++) {
    const a = r * (SEG + 1) + j, c = a + SEG + 1;
    index.push(a, a + 1, c, a + 1, c + 1, c);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  geo.setAttribute("skinIndex", new THREE.Uint16BufferAttribute(skinIndex, 4));
  geo.setAttribute("skinWeight", new THREE.BufferAttribute(skinWeight, 4));
  geo.setIndex(index);
  geo.computeVertexNormals();

  let prev = null;
  const bones = boneS.map((s, i) => {
    const bone = new THREE.Bone();
    bone.name = `Bone_${String(i + 1).padStart(2, "0")}`;
    bone.position.set(0, 0, i ? s - boneS[i - 1] : s);
    if (prev) prev.add(bone);
    prev = bone;
    return bone;
  });
  const root = new THREE.Group();
  root.name = "SnakeRoot";
  const mesh = new THREE.SkinnedMesh(geo, p.skin);
  mesh.name = "Body";
  mesh.castShadow = mesh.receiveShadow = true;
  mesh.frustumCulled = false;
  mesh.add(bones[0]);
  root.add(mesh);

  const head = bones[0], at = (s) => { const [rx, ry] = profile(s, L); return { rx, ry, cy: ry * 0.8 }; };
  const local = (x, y, s) => new THREE.Vector3(x, y, s - boneS[0]);
  const e = at(0.25);
  for (const side of [1, -1]) {
    const eye = new THREE.Mesh(p.eyeGeo, p.eye);
    eye.name = "Eye"; eye.scale.setScalar(0.044);
    eye.position.copy(local(side * (e.rx * 0.917 - 0.014), e.cy + e.ry * 0.4, 0.25));
    const pupil = new THREE.Mesh(p.eyeGeo, p.pupil);
    pupil.name = "Pupil"; pupil.scale.set(0.24, 0.62, 0.24);
    pupil.position.set(side * 0.78, 0.16, -0.46).normalize().multiplyScalar(0.8);
    eye.add(pupil);
    head.add(eye);
    const n = at(0.05), nostril = new THREE.Mesh(p.eyeGeo, p.dark);
    nostril.name = "Nostril"; nostril.scale.setScalar(0.011);
    nostril.position.copy(local(side * n.rx * 0.45, n.cy + n.ry * 0.86, 0.05));
    head.add(nostril);
    const lip = [];
    for (let s = 0.03; s <= 0.32; s += 0.03) { const q = at(s); lip.push(local(side * q.rx * 0.985, q.cy - q.ry * 0.28, s)); }
    const mouth = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(lip), 16, 0.0045, 4), p.dark);
    mouth.name = "Mouth";
    head.add(mouth);
  }
  const tongue = makeTongue();
  const t0 = at(0.02);
  tongue.position.copy(local(0, t0.cy - t0.ry * 0.2, 0.03));
  head.add(tongue);

  root.updateMatrixWorld(true);
  mesh.bind(new THREE.Skeleton(bones));
  return { root, mesh, chain: bones, tongue };
}

// The longest bone chain in a skeleton, root first.
function longestChain(bones) {
  const set = new Set(bones);
  let best = [];
  const walk = (b, path) => {
    path.push(b);
    const kids = b.children.filter((c) => set.has(c));
    if (!kids.length && path.length > best.length) best = path.slice();
    kids.forEach((k) => walk(k, path));
    path.pop();
  };
  bones.filter((b) => !set.has(b.parent)).forEach((r) => walk(r, []));
  return best;
}

// A snake model driven by the smoothed path: its bones are laid along the centre line every frame.
export class Snake3D {
  constructor(parent) {
    this.group = new THREE.Group();
    this.group.name = "snake";
    parent.add(this.group);
    this.kind = "none";
    this.rig = null;
    this.mixer = null;
    this.tongueAction = null;
    this.headFrame = new THREE.Object3D();
    this.group.add(this.headFrame);
    this.lastHead = new THREE.Vector3(Infinity, 0, 0);
    this.nextFlick = 1500;
  }

  // Loads `snake.glb`; falls back to the built-in rigged model when the file is missing or unusable.
  async load(url = SNAKE_URL) {
    try {
      const gltf = await new GLTFLoader().loadAsync(url);
      this.useGLTF(gltf);
      this.kind = "glb";
    } catch (err) {
      if (!String(err).includes("404")) console.warn("[snake3d] snake.glb unusable, using built-in model:", err);
      this.rebuildProcedural(8);
      this.kind = "procedural";
    }
    return this.kind;
  }

  // Wraps a loaded glTF: scales it to the board, finds its bone chain and head end, and adopts its animations.
  useGLTF(gltf) {
    const model = gltf.scene;
    let skinned = null;
    model.traverse((o) => {
      if (o.isSkinnedMesh && !skinned) skinned = o;
      if (o.isMesh) { o.castShadow = o.receiveShadow = true; o.frustumCulled = false; }
    });
    if (!skinned) throw new Error("snake.glb has no skinned mesh");
    let chain = longestChain(skinned.skeleton.bones);
    if (chain.length < 4) throw new Error("snake.glb bone chain is too short");
    const names = (bs) => bs.map((b) => b.name.toLowerCase()).join(" ");
    if (/head|skull|jaw/.test(names(chain.slice(-3))) || /tail/.test(names(chain.slice(0, 3)))) chain = chain.slice().reverse();
    model.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(model);
    model.scale.multiplyScalar((2 * RADIUS * 0.86) / Math.max(1e-6, box.max.y - box.min.y));
    this.group.add(model);
    const clip = gltf.animations.find((a) => /tongue|flick/i.test(a.name));
    let tongue = null;
    if (clip) {
      this.mixer = new THREE.AnimationMixer(model);
      this.tongueAction = this.mixer.clipAction(clip);
      this.tongueAction.setLoop(THREE.LoopOnce, 1);
    } else {
      model.traverse((o) => { if (/tongue/i.test(o.name)) o.visible = false; });
      tongue = makeTongue();
      this.headFrame.add(tongue);
    }
    this.setupRig({ root: model, mesh: skinned, chain, tongue }, !!tongue);
  }

  // Replaces the procedural model with one of rest length `L`, disposing the old one.
  rebuildProcedural(L) {
    if (this.rig) {
      this.group.remove(this.rig.root);
      this.rig.mesh.geometry.dispose();
      this.rig.mesh.skeleton.dispose();
      this.rig.root.traverse((o) => { if (o.name === "Mouth") o.geometry.dispose(); });
    }
    this.group.add((this.built = buildProceduralSnake(L)).root);
    this.setupRig(this.built, false);
  }

  // Records each chain bone's rest placement relative to the body so it can be re-posed along any path.
  setupRig({ root, mesh, chain, tongue }, tongueOnFrame) {
    root.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(mesh);
    const n = chain.length, pos = chain.map((b) => new THREE.Vector3().setFromMatrixPosition(b.matrixWorld));
    const fwd = pos.map((p, i) => {
      const a = pos[Math.max(0, i - 1)], c = pos[Math.min(n - 1, i + 1)];
      return new THREE.Vector3(a.x - c.x, 0, a.z - c.z).normalize();
    });
    const corners = [];
    for (let i = 0; i < 8; i++) corners.push(new THREE.Vector3(i & 1 ? box.max.x : box.min.x, i & 2 ? box.max.y : box.min.y, i & 4 ? box.max.z : box.min.z));
    const lead = Math.max(...corners.map((c) => _p.subVectors(c, pos[0]).dot(fwd[0])));
    const trail = Math.max(...corners.map((c) => -_p.subVectors(c, pos[n - 1]).dot(fwd[n - 1])));
    const restS = new Float32Array(n);
    restS[0] = Math.max(0, lead);
    for (let i = 1; i < n; i++) restS[i] = restS[i - 1] + pos[i].distanceTo(pos[i - 1]);
    const bones = chain.map((bone, i) => {
      const q = new THREE.Quaternion(), scale = new THREE.Vector3();
      bone.matrixWorld.decompose(_p, q, scale);
      const off = bodyFrame(fwd[i], new THREE.Quaternion()).invert().multiply(q);
      return { bone, off, scale, height: pos[i].y - box.min.y, cur: new THREE.Quaternion(), fwd: fwd[i].clone(), world: new THREE.Matrix4() };
    });
    const index = new Map(chain.map((b, i) => [b, i]));
    const depth = (b) => { let d = 0; for (let p = b.parent; p; p = p.parent) d++; return d; };
    const order = chain.map((b, i) => i).sort((a, b) => depth(chain[a]) - depth(chain[b]));
    const outerParentInv = chain.map((b) => index.has(b.parent) ? null : new THREE.Matrix4().copy(b.parent.matrixWorld).invert());
    this.rig = {
      root, mesh, bones, order, parentIdx: chain.map((b) => index.has(b.parent) ? index.get(b.parent) : -1), outerParentInv,
      restS, restLen: restS[n - 1] + Math.max(0.05, trail), tongue, tongueOnFrame, headLift: bones[0].height,
    };
    if (tongueOnFrame) tongue.position.set(0, 0.1 - bones[0].height, 0.03 - restS[0]);
    this.snapNext = true;
  }

  // Writes the path point `d` cells behind the head into `out`, extending straight past either end.
  pathPoint(cl, d, dir, out) {
    if (cl.n < 2) { out.x = cl.x[0] - dir[0] * d; out.y = cl.y[0] - dir[1] * d; return out; }
    if (d < 0 || d > cl.total) {
      const head = d < 0, base = head ? 0 : cl.total;
      cl.pointAt(base, _e);
      cl.pointAt(head ? Math.min(0.2, cl.total) : Math.max(0, cl.total - 0.2), out);
      let dx = _e.x - out.x, dy = _e.y - out.y;
      const l = Math.hypot(dx, dy) || 1; dx /= l; dy /= l;
      const over = head ? -d : d - cl.total;
      out.x = _e.x + dx * over; out.y = _e.y + dy * over;
      return out;
    }
    return cl.pointAt(d, out);
  }

  // Poses the snake on this frame's centre line (cell units), plus its tongue and animations.
  update(cl, state, now, dt, rebuilt = false) {
    const rig = this.rig;
    if (!rig || !cl.n) return;
    const W = state.width, H = state.height;
    const dir = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] }[state.direction] || [1, 0];
    const target = cl.total + TIP_AHEAD + TAIL_BEHIND;
    let k = (target - NECK) / Math.max(0.05, rig.restLen - NECK);
    if (this.kind === "procedural" && !rebuilt && (k > 1.3 || (k < 0.5 && rig.restLen > 3.5))) {
      const cells = state.width * state.height + 4;
      this.rebuildProcedural(Math.max(3, Math.min(cells, Math.round(target * 1.1 * 2) / 2)));
      return this.update(cl, state, now, dt, true);
    }
    k = Math.max(0.05, k);

    const hx = cl.x[0] - W / 2, hz = cl.y[0] - H / 2;
    const jumped = Math.hypot(hx - this.lastHead.x, hz - this.lastHead.z) > 1.5;
    this.lastHead.set(hx, 0, hz);
    const snap = this.snapNext || jumped;
    this.snapNext = false;
    const alpha = snap ? 1 : 1 - Math.exp(-dt / 35);

    for (let i = 0; i < rig.bones.length; i++) {
      const bd = rig.bones[i], s = rig.restS[i];
      const d = -TIP_AHEAD + (s <= NECK ? s : NECK + (s - NECK) * k);
      this.pathPoint(cl, d, dir, _a);
      this.pathPoint(cl, d - 0.14, dir, _b);
      this.pathPoint(cl, d + 0.14, dir, _c);
      _f.set(_b.x - _c.x, 0, _b.y - _c.y);
      if (_f.lengthSq() > 1e-8) bd.fwd.copy(_f.normalize());
      bodyFrame(bd.fwd, _q).multiply(bd.off);
      if (snap) bd.cur.copy(_q); else bd.cur.slerp(_q, alpha);
      _p.set(_a.x - W / 2, bd.height, _a.y - H / 2);
      bd.world.compose(_p, bd.cur, bd.scale);
      if (i === 0) { this.headFrame.position.copy(_p); bodyFrame(bd.fwd, this.headFrame.quaternion); }
    }
    for (const i of rig.order) {
      const bd = rig.bones[i], pi = rig.parentIdx[i];
      if (pi >= 0) _inv.copy(rig.bones[pi].world).invert(); else _inv.copy(rig.outerParentInv[i]);
      _m.multiplyMatrices(_inv, bd.world);
      _m.decompose(bd.bone.position, bd.bone.quaternion, bd.bone.scale);
    }
    this.animateTongue(now, dt, state.game_over);
  }

  // Flicks the tongue every few seconds; it stays in once the round is over.
  animateTongue(now, dt, over) {
    if (this.mixer) {
      this.mixer.update(dt / 1000);
      if (!over && now > this.nextFlick) { this.tongueAction.reset().play(); this.nextFlick = now + 2600 + Math.random() * 1800; }
      return;
    }
    const t = this.rig.tongue;
    if (!t) return;
    if (over) { t.visible = false; this.flickAt = -1; return; }
    if (now > this.nextFlick) { this.flickAt = now; this.nextFlick = now + 2400 + Math.random() * 2200; }
    const p = this.flickAt >= 0 ? (now - this.flickAt) / 420 : 1;
    if (p >= 1) { t.visible = false; return; }
    const out = Math.sin(p * Math.PI);
    t.visible = true;
    t.scale.set(1, 1, Math.max(0.01, out));
    t.rotation.x = Math.sin(p * Math.PI * 6) * 0.14 * out;
  }

  // Shows where the model came from, for the debug panel.
  describe() {
    return this.kind === "glb" ? `snake.glb · ${this.rig.bones.length} bones` : `built-in rig · ${this.rig ? this.rig.bones.length : 0} bones`;
  }

  dispose() {
    if (this.rig && this.kind === "procedural") { this.rig.mesh.geometry.dispose(); this.rig.mesh.skeleton.dispose(); }
    this.group.removeFromParent();
  }
}
