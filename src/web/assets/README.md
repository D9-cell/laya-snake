# 3D assets (optional)

The browser dashboard's 3D view loads these files if they exist. Nothing here is required: a missing
or unusable file falls back to the built-in models, and the game keeps running either way.

The server reads this folder from disk at request time (`$LAYA_WEB_ASSETS`, else `src/web/assets`
under the working directory, else `assets/` next to the binary), so you can drop a file in and
reload the page without rebuilding.

| File | What it is |
|---|---|
| `snake.glb` | The snake. Must contain a `SkinnedMesh` with a chain of bones running head to tail. |
| `rabbit.glb` | The food: a sleeping rabbit. Any static or animated glTF works. |
| `environment/` | Reserved for future environment models and textures. |

## `snake.glb` requirements

- Rest (bind) pose lying **straight and flat** along any horizontal axis, back facing up (+Y).
- One bone chain from head to tail (`SnakeRoot > Bone_01 > Bone_02 > ...` or similar), with the
  body skinned smoothly along it. The longest chain in the skeleton is used. The head end is taken to
  be the chain's root unless the bones near the leaf are named `head`/`skull`/`jaw` (or those near
  the root are named `tail`).
- Any scale: the model is scaled so its body height matches the board (about half a cell).
- Denser bones bend more smoothly. The body is stretched along the path to the snake's current
  length; the first 0.62 cells behind the snout (the head) never stretch.
- An animation clip whose name contains `tongue` or `flick` is played every few seconds. Without
  one, any node named `tongue` is hidden and a built-in forked tongue is animated instead.
- Materials are used as they are (PBR metallic-roughness from glTF).
