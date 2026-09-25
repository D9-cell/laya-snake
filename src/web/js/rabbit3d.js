import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { clone as cloneSkinned } from "three/addons/utils/SkeletonUtils.js";
import { furTextures } from "./textures3d.js";
import { hash } from "./util.js";

export const RABBIT_URL = "/assets/rabbit.glb";

const COATS = [0xe9e2d6, 0xcdb08c, 0x9d7b5a, 0x8b847c];

let zTexture = null;
// A soft "z" glyph for the sleeping indicator (drawn once).
function zSprite() {
  if (!zTexture) {
    const cv = document.createElement("canvas");
    cv.width = cv.height = 64;
    const ctx = cv.getContext("2d");
    ctx.font = "600 48px ui-sans-serif, system-ui, sans-serif";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillStyle = "#ffffff"; ctx.fillText("z", 32, 34);
    zTexture = new THREE.CanvasTexture(cv);
    zTexture.colorSpace = THREE.SRGBColorSpace;
  }
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: zTexture, transparent: true, depthWrite: false, opacity: 0 }));
  s.scale.setScalar(0.14);
  return s;
}

// A curled-up sleeping rabbit about 0.6 cells long, built from shared geometry: body, haunch,
// tucked head, ears laid flat along the back, tail, closed eyes and nose. Faces -Z.
function buildRabbitTemplate(coat) {
  const fur = furTextures();
  const furMat = new THREE.MeshPhysicalMaterial({
    color: coat, roughness: 0.95, normalMap: fur.normalMap, normalScale: new THREE.Vector2(0.5, 0.5),
    sheen: 1, sheenRoughness: 0.75, sheenColor: new THREE.Color(0xffffff),
  });
  const pale = new THREE.MeshPhysicalMaterial({ color: 0xf4efe6, roughness: 1, sheen: 1, sheenRoughness: 0.8, sheenColor: new THREE.Color(0xffffff) });
  const pink = new THREE.MeshStandardMaterial({ color: 0xc9a09a, roughness: 0.8 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x1d1512, roughness: 0.5 });
  const sphere = new THREE.SphereGeometry(1, 28, 20);
  const g = new THREE.Group();
  const add = (geo, mat, [x, y, z], [sx, sy, sz], rot = [0, 0, 0]) => {
    const m = new THREE.Mesh(geo, mat);
    m.position.set(x, y, z); m.scale.set(sx, sy, sz); m.rotation.set(...rot);
    m.castShadow = true; m.receiveShadow = true;
    g.add(m);
    return m;
  };
  const body = add(sphere, furMat, [0, 0.14, 0.05], [0.18, 0.14, 0.24]);
  body.name = "Body";
  add(sphere, furMat, [0, 0.12, 0.14], [0.175, 0.12, 0.15]);
  const head = add(sphere, furMat, [0, 0.14, -0.24], [0.1, 0.09, 0.11]);
  add(sphere, pale, [0, 0.11, -0.315], [0.05, 0.04, 0.035]);
  const ear = new THREE.CapsuleGeometry(1, 2.6, 6, 12);
  for (const s of [1, -1]) {
    add(ear, furMat, [s * 0.042, 0.262, -0.1], [0.03, 0.05, 0.02], [Math.PI / 2 - 0.06, 0, s * 0.14]);
    add(ear, pink, [s * 0.05, 0.272, -0.1], [0.013, 0.04, 0.008], [Math.PI / 2 - 0.06, 0, s * 0.14]);
    add(new THREE.TorusGeometry(0.017, 0.0035, 5, 10, Math.PI), dark, [s * 0.085, 0.16, -0.29], [1, 1, 1], [0, s * 1.2, Math.PI]);
  }
  add(sphere, pink, [0, 0.12, -0.347], [0.013, 0.01, 0.009]);
  add(sphere, pale, [0, 0.14, 0.3], [0.05, 0.05, 0.045]);
  g.userData.breathe = body;
  g.userData.head = head;
  return g;
}

// Pool of rabbits, one per food cell reported by the server.
export class Rabbits {
  constructor(parent) {
    this.group = new THREE.Group();
    this.group.name = "food";
    parent.add(this.group);
    this.templates = [];
    this.animated = false;
    this.clips = [];
    this.active = new Map();
    this.pool = [];
    this.kind = "none";
    this.showZ = true;
  }

  // Loads `rabbit.glb` (scaled to about 0.6 cells) or falls back to the built-in rabbit.
  async load(url = RABBIT_URL) {
    try {
      const gltf = await new GLTFLoader().loadAsync(url);
      const model = gltf.scene;
      model.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(model), size = box.getSize(new THREE.Vector3());
      model.scale.multiplyScalar(0.6 / Math.max(size.x, size.z, 1e-6));
      model.updateMatrixWorld(true);
      const box2 = new THREE.Box3().setFromObject(model);
      model.position.y -= box2.min.y;
      model.traverse((o) => { if (o.isMesh) { o.castShadow = o.receiveShadow = true; } });
      const holder = new THREE.Group();
      holder.add(model);
      this.templates = [holder];
      this.clips = gltf.animations;
      this.animated = this.clips.length > 0;
      this.kind = "glb";
    } catch (err) {
      if (!String(err).includes("404")) console.warn("[rabbit3d] rabbit.glb unusable, using built-in model:", err);
      this.templates = COATS.map(buildRabbitTemplate);
      this.kind = "procedural";
    }
    return this.kind;
  }

  // Makes a new rabbit instance from a template, sharing its geometry and materials.
  spawn(variant) {
    const tpl = this.templates[variant % this.templates.length];
    const r = new THREE.Group();
    const body = this.kind === "glb" ? cloneSkinned(tpl) : tpl.clone(true);
    r.add(body);
    r.userData.body = body;
    r.userData.breathe = this.kind === "glb" ? body : body.getObjectByName("Body");
    r.userData.variant = variant;
    if (this.animated) {
      r.userData.mixer = new THREE.AnimationMixer(body);
      r.userData.mixer.clipAction(this.clips[0]).play();
    }
    r.userData.zs = [0, 1, 2].map(() => { const s = zSprite(); r.add(s); return s; });
    return r;
  }

  // Matches rabbits to the server's food cells: kept foods stay put, eaten ones leave, new ones settle in.
  setFoods(foods, W, H, now) {
    const keep = new Set();
    for (const [x, y] of foods) {
      const key = x + "," + y;
      keep.add(key);
      if (this.active.has(key)) continue;
      const variant = Math.floor(hash(x, y, 19) * 4);
      let r = this.pool.findIndex((p) => p.userData.variant % this.templates.length === variant % this.templates.length);
      r = r >= 0 ? this.pool.splice(r, 1)[0] : this.spawn(variant);
      r.position.set(x + 0.5 - W / 2, 0, y + 0.5 - H / 2);
      r.rotation.y = hash(x, y, 23) * Math.PI * 2;
      r.userData.seed = hash(x, y, 29) * 10;
      r.userData.born = now;
      r.visible = true;
      this.group.add(r);
      this.active.set(key, r);
    }
    for (const [key, r] of this.active) {
      if (keep.has(key)) continue;
      this.active.delete(key);
      this.group.remove(r);
      this.pool.push(r);
    }
  }

  // Breathing, settling-in and the drifting "z"s.
  update(now, dt) {
    for (const r of this.active.values()) {
      const u = r.userData, grow = Math.min(1, (now - u.born) / 380);
      r.scale.setScalar(grow * (2 - grow));
      u.breathe.scale.y = (this.kind === "glb" ? 1 : 0.14) * (1 + 0.035 * Math.sin(now / 620 + u.seed));
      if (u.mixer) u.mixer.update(dt / 1000);
      u.zs.forEach((z, k) => {
        const ph = ((now / 1000 + u.seed + k * 0.9) % 2.7) / 2.7;
        z.visible = this.showZ;
        z.material.opacity = Math.sin(ph * Math.PI) * 0.55;
        z.position.set(0.08 + ph * 0.12, 0.34 + ph * 0.32, -0.12);
        z.scale.setScalar(0.07 + ph * 0.06);
      });
    }
  }

  // Forgets every rabbit (used when the board changes size).
  clear() {
    for (const r of this.active.values()) { this.group.remove(r); this.pool.push(r); }
    this.active.clear();
  }

  dispose() {
    this.clear();
    this.group.removeFromParent();
  }
}
