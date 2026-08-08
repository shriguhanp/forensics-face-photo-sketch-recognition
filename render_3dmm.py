"""
render_3dmm.py

Renders synthetic face images directly from Basel Face Model 2019 (BFM 2019)
using a pure Python/NumPy/Matplotlib software renderer.

NO Java, NO OpenGL display, NO Basel Illumination Prior, NO DTD backgrounds needed.
Uses only model2019_bfm.h5 which you already have.

Strategy:
  - Loads the BFM 2019 shape + color + expression PCA spaces from HDF5
  - For each CUFS test identity, samples K random face instances by perturbing
    shape/color/expression coefficients with controlled variance
  - Projects to 2D orthographically with a random small yaw rotation per variant
  - Renders using painter's algorithm (depth sort) + matplotlib Agg (headless)
  - Adds simple diffuse lighting so faces look 3D
  - Saves 160x160 PNG images + writes manifest CSV for import_3dmm_synthetics.py

Run:
    python3 render_3dmm.py --k 9 --output_dir 3dmm_renders --manifest my_3dmm_renders.csv
    python3 import_3dmm_synthetics.py --manifest my_3dmm_renders.csv
    python3 pipeline.py evaluate --mode forensic
"""

import argparse
import json
import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')           # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.collections as mc

IMAGE_SIZE   = 160
BFM_PATH     = "model2019_bfm.h5"
TEST_PAIRS   = "test_pairs.json"

# How many PCA components to use (fewer = more "average" looking, stable)
N_SHAPE_COMPONENTS  = 40   # out of 199
N_COLOR_COMPONENTS  = 30   # out of 199
N_EXPR_COMPONENTS   = 15   # out of 100

# Coefficient std-dev scale: 1.0 = sample from the model prior normally,
# lower = closer to mean face, higher = more extreme variation
SHAPE_SCALE  = 0.6
COLOR_SCALE  = 0.5
EXPR_SCALE   = 0.4

# Subsample the mesh for speed (every Nth triangle). 1 = all 94k triangles (slow).
# 4 = ~23k triangles, looks good at 160x160 and renders in a few seconds per image.
MESH_STRIDE = 4


# ---------------------------------------------------------------------------
# BFM 2019 loader
# ---------------------------------------------------------------------------

class BFM2019:
    def __init__(self, h5_path):
        try:
            import h5py
        except ImportError:
            print("ERROR: h5py not installed. Run: pip install h5py")
            sys.exit(1)

        print(f"Loading BFM 2019 from {h5_path} ...")
        with h5py.File(h5_path, 'r') as f:
            self.shape_mean  = f['shape/model/mean'][:].astype(np.float32)
            self.shape_basis = f['shape/model/pcaBasis'][:, :N_SHAPE_COMPONENTS].astype(np.float32)
            self.shape_std   = np.sqrt(f['shape/model/pcaVariance'][:N_SHAPE_COMPONENTS]).astype(np.float32)

            self.color_mean  = f['color/model/mean'][:].astype(np.float32)
            self.color_basis = f['color/model/pcaBasis'][:, :N_COLOR_COMPONENTS].astype(np.float32)
            self.color_std   = np.sqrt(f['color/model/pcaVariance'][:N_COLOR_COMPONENTS]).astype(np.float32)

            self.expr_mean   = f['expression/model/mean'][:].astype(np.float32)
            self.expr_basis  = f['expression/model/pcaBasis'][:, :N_EXPR_COMPONENTS].astype(np.float32)
            self.expr_std    = np.sqrt(f['expression/model/pcaVariance'][:N_EXPR_COMPONENTS]).astype(np.float32)

            # Triangles: stored (3, F) — transpose to (F, 3)
            cells_full       = f['shape/representer/cells'][:].T.astype(np.int32)

        # Subsample triangles for speed
        self.cells = cells_full[::MESH_STRIDE]
        self.n_vertices = self.shape_mean.shape[0] // 3
        print(f"  Vertices: {self.n_vertices:,}  |  Triangles used: {len(self.cells):,} "
              f"(of {len(cells_full):,}, stride={MESH_STRIDE})")

    def make_instance(self, rng):
        """
        Sample one random face instance. Returns:
            vertices: (N, 3) float32 in mm
            colors:   (N, 3) float32 in [0, 1]
        """
        s_coef = rng.standard_normal(N_SHAPE_COMPONENTS).astype(np.float32) * self.shape_std * SHAPE_SCALE
        c_coef = rng.standard_normal(N_COLOR_COMPONENTS).astype(np.float32) * self.color_std * COLOR_SCALE
        e_coef = rng.standard_normal(N_EXPR_COMPONENTS).astype(np.float32) * self.expr_std * EXPR_SCALE

        verts  = (self.shape_mean + self.shape_basis @ s_coef
                                  + self.expr_mean   + self.expr_basis @ e_coef).reshape(-1, 3)
        # Color is already stored in [0, 1] in BFM 2019 (not [0, 255])
        colors = np.clip((self.color_mean + self.color_basis @ c_coef).reshape(-1, 3),
                         0.0, 1.0)
        return verts, colors


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def _rot_y(angle_rad):
    """Rotation matrix around Y axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def _rot_x(angle_rad):
    """Rotation matrix around X axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def _compute_face_normals(verts, triangles):
    """Compute per-face unit normals."""
    v0 = verts[triangles[:, 0]]
    v1 = verts[triangles[:, 1]]
    v2 = verts[triangles[:, 2]]
    n  = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return n / norms


def render_face(model, verts, colors, yaw_deg=0.0, pitch_deg=5.0, img_size=IMAGE_SIZE):
    """
    Render one face instance to a PIL-compatible uint8 numpy array (H, W, 3).

    yaw_deg:   horizontal rotation in degrees (positive = turn right)
    pitch_deg: vertical tilt in degrees (slight downward look)
    """
    # --- Apply rotation -------------------------------------------------------
    R = _rot_y(np.deg2rad(yaw_deg)) @ _rot_x(np.deg2rad(pitch_deg))
    v = (R @ verts.T).T           # (N, 3)

    # --- Orthographic projection: x→screen_x, -y→screen_y -------------------
    x = v[:, 0]
    y = v[:, 1]
    z = v[:, 2]

    margin = 0.08
    x_range = x.ptp()
    y_range = y.ptp()
    scale   = (1.0 - 2 * margin) * img_size / max(x_range, y_range, 1e-6)

    px = (x - x.mean()) * scale + img_size / 2.0   # screen x (right)
    py = -(y - y.mean()) * scale + img_size / 2.0  # screen y (down, flip)

    screen = np.stack([px, py], axis=1)             # (N, 2)

    # --- Per-face depth + visibility + shading --------------------------------
    tri = model.cells                               # (F, 3)
    F   = len(tri)

    # Mean z of each triangle (for painter's sort — paint far first)
    mean_z = z[tri].mean(axis=1)                   # (F,)

    # Per-face color = mean of vertex colors
    face_colors = colors[tri].mean(axis=1)         # (F, 3)

    # Simple diffuse shading: light from slightly upper-left front
    light_dir  = np.array([0.3, 0.5, 1.0], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)

    normals    = _compute_face_normals(v, tri)     # (F, 3)
    diffuse    = np.clip(normals @ light_dir, 0, 1)  # (F,)
    ambient    = 0.35

    # Backface cull: skip triangles whose normal z-component points away
    front_mask = normals[:, 2] > -0.1             # (F,)

    shaded_colors = np.clip(face_colors * (ambient + (1 - ambient) * diffuse[:, None]), 0, 1)

    # --- Painter's sort (back to front) ----------------------------------------
    order = np.argsort(mean_z)                     # ascending z → paint far first

    # --- Matplotlib Agg rendering (headless, C-backed) -------------------------
    dpi = img_size   # so figure size = 1 inch = img_size px
    fig = plt.figure(figsize=(1, 1), dpi=dpi)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, img_size)
    ax.set_ylim(img_size, 0)    # screen coords (y down)
    ax.axis('off')
    ax.set_facecolor('white')

    # Build triangle vertex arrays in painter's order
    vis_idx      = order[front_mask[order]]         # ordered, front-only indices
    tri_verts_2d = screen[tri[vis_idx]]             # (Fvis, 3, 2)
    tri_cols     = shaded_colors[vis_idx]           # (Fvis, 3)

    poly = mc.PolyCollection(tri_verts_2d,
                             facecolors=tri_cols,
                             edgecolors='none',
                             linewidths=0)
    ax.add_collection(poly)

    fig.canvas.draw()
    # Support both older (tostring_rgb) and newer (buffer_rgba) matplotlib APIs
    try:
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = buf.reshape(img_size, img_size, 3)
    except AttributeError:
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = buf.reshape(img_size, img_size, 4)[:, :, :3].copy()
    plt.close(fig)
    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bfm',        default=BFM_PATH,  help='Path to model2019_bfm.h5')
    parser.add_argument('--test_pairs', default=TEST_PAIRS, help='Path to test_pairs.json')
    parser.add_argument('--k',   type=int, default=9,       help='Variants per identity (default 9)')
    parser.add_argument('--output_dir', default='3dmm_renders', help='Output directory for renders')
    parser.add_argument('--manifest',   default='my_3dmm_renders.csv', help='Output CSV manifest')
    parser.add_argument('--seed', type=int, default=42,     help='Global random seed')
    args = parser.parse_args()

    # --- Validate inputs -------------------------------------------------------
    if not os.path.exists(args.bfm):
        print(f"ERROR: BFM file not found: {args.bfm}")
        print("Make sure model2019_bfm.h5 is in the project directory.")
        sys.exit(1)
    if not os.path.exists(args.test_pairs):
        print(f"ERROR: {args.test_pairs} not found. Run prepare_data.py first.")
        sys.exit(1)

    with open(args.test_pairs) as f:
        test_identities = json.load(f)

    print(f"Generating {args.k} 3DMM variants for {len(test_identities)} test identities "
          f"→ {args.k * len(test_identities)} total images")

    # --- Load model ------------------------------------------------------------
    model = BFM2019(args.bfm)

    # --- Output dir ------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Render ----------------------------------------------------------------
    manifest_rows = []
    rng_global    = np.random.default_rng(args.seed)

    # Yaw angles to sample across K variants — spread across ±20°
    # Variant 0 is frontal, others add pose diversity
    def yaw_for_variant(k, K):
        if K == 1:
            return 0.0
        # Half frontal, half with gentle yaw variation
        angles = [0.0] + list(np.linspace(-20, 20, K - 1))
        return angles[k % len(angles)]

    for entry in test_identities:
        identity_id = entry['identity_id']
        id_dir      = os.path.join(args.output_dir, str(identity_id))
        os.makedirs(id_dir, exist_ok=True)

        # Per-identity seed so each identity gets different face shape
        id_rng = np.random.default_rng(args.seed + identity_id * 1000)

        for k in range(args.k):
            # Different expression / color per variant, same shape identity
            variant_rng = np.random.default_rng(args.seed + identity_id * 1000 + k)
            verts, colors = model.make_instance(variant_rng)

            yaw   = yaw_for_variant(k, args.k)
            pitch = variant_rng.uniform(-5, 5)

            img   = render_face(model, verts, colors, yaw_deg=yaw, pitch_deg=pitch)

            out_path = os.path.join(id_dir, f"{identity_id}_{k}.png")
            from PIL import Image
            Image.fromarray(img).save(out_path)
            manifest_rows.append({'identity_id': identity_id, 'image_path': out_path})

        print(f"  identity {identity_id:3d}: {args.k} variants saved to {id_dir}/")

    # --- Write manifest --------------------------------------------------------
    import csv
    with open(args.manifest, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['identity_id', 'image_path'])
        w.writeheader()
        w.writerows(manifest_rows)

    print(f"\nDone. {len(manifest_rows)} images written.")
    print(f"Manifest: {args.manifest}")
    print(f"\nNext steps:")
    print(f"  python3 import_3dmm_synthetics.py --manifest {args.manifest}")
    print(f"  python3 pipeline.py evaluate --mode forensic")


if __name__ == '__main__':
    main()
