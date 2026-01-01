import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import torch
import smplx
import pyrender
import trimesh
from PIL import Image

# =============================
# SETTINGS
# =============================
NPZ_FOLDER = "/content/drive/MyDrive/Extracted_Parameters"
MODEL_PATH = "/content/drive/MyDrive/OSXBackup/OSX/common/utils/human_model_files/smplx/SMPLX_NEUTRAL.npz"
OUT_FRAMES = "/content/drive/MyDrive/rendered_frames"
PERSON_ID = 0

os.makedirs(OUT_FRAMES, exist_ok=True)

H, W = 720, 1280

UPPER_FRAC = 0.55
PAD_X = 0.50
PAD_Y = 0.14
BG = (245, 245, 245, 255)

def project_uv(verts_cam, fx, fy, cx, cy):
    X, Y, Z = verts_cam[:, 0], verts_cam[:, 1], verts_cam[:, 2]
    return np.stack([fx*(X/Z) + cx, fy*(Y/Z) + cy], axis=1)

# Load SMPL-X
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smplx.create(
    model_path=MODEL_PATH, model_type="smplx", gender="neutral",
    num_betas=10, num_expression_coeffs=10, use_pca=False,
    use_face_contour=True, ext="npz"
).to(device).eval()

files = sorted([f for f in os.listdir(NPZ_FOLDER) if f.endswith(".npz")])
print("[INFO] Total frames:", len(files))

for idx, fname in enumerate(files):
    print(f"[INFO] Rendering frame {idx+1}/{len(files)}: {fname}")

    Z = np.load(os.path.join(NPZ_FOLDER, fname), allow_pickle=False)
    p = f"person_{PERSON_ID}_smplx_"

    params_np = {
        "global_orient":  Z[p+"root_pose"].reshape(1,3).astype(np.float32),
        "body_pose":      Z[p+"body_pose"].reshape(1,-1).astype(np.float32),
        "left_hand_pose": Z[p+"lhand_pose"].reshape(1,-1).astype(np.float32),
        "right_hand_pose":Z[p+"rhand_pose"].reshape(1,-1).astype(np.float32),
        "jaw_pose":       Z[p+"jaw_pose"].reshape(1,3).astype(np.float32),
        "betas":          Z[p+"shape"].reshape(1,-1).astype(np.float32),
        "expression":     Z[p+"expr"].reshape(1,-1).astype(np.float32),
        "leye_pose":      np.zeros((1,3), dtype=np.float32),
        "reye_pose":      np.zeros((1,3), dtype=np.float32),
    }

    cam_trans = Z[f"person_{PERSON_ID}_cam_trans"].astype(np.float32)
    fx, fy = map(float, Z[f"person_{PERSON_ID}_focal"])
    cx, cy = map(float, Z[f"person_{PERSON_ID}_princpt"])

    with torch.no_grad():
        tens = {k: torch.tensor(v, dtype=torch.float32).to(device)
                for k, v in params_np.items()}
        verts = model(**tens).vertices[0].cpu().numpy()
    faces = model.faces

    verts_cam = verts + cam_trans[None, :]

    # =============================
    # 1) AUTO-FIT FULL BODY (CRITICAL)
    # =============================
    uv = project_uv(verts_cam, fx, fy, cx, cy)
    umin, vmin = uv.min(axis=0)
    umax, vmax = uv.max(axis=0)
    bbox_w, bbox_h = umax - umin, vmax - vmin

    W_eff, H_eff = W*0.9, H*0.9
    k = max(bbox_w / W_eff, bbox_h / H_eff, 1.0)
    verts_cam[:,2] *= k  # push backward

    # =============================
    # 2) RECENTER MESH (CRITICAL)
    # =============================
    uv = project_uv(verts_cam, fx, fy, cx, cy)
    uc, vc = uv.mean(axis=0)
    du = (W/2 - uc)
    dv = (H/2 - vc)
    z_mean = np.median(verts_cam[:,2])

    verts_cam[:,0] += (du * z_mean) / fx
    verts_cam[:,1] += (dv * z_mean) / fy

    # =============================
    # Render
    # =============================
    verts_gl = verts_cam.copy()
    verts_gl[:,1] *= -1
    verts_gl[:,2] *= -1

    scene = pyrender.Scene(bg_color=[c/255 for c in BG])
    mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.86,0.86,0.86,1], metallicFactor=0.0, roughnessFactor=0.7
    )
    tri = trimesh.Trimesh(verts_gl, faces, process=False)
    scene.add(pyrender.Mesh.from_trimesh(tri, material=mat, smooth=True))
    scene.add(pyrender.IntrinsicsCamera(fx, fy, cx, cy), pose=np.eye(4))
    scene.add(pyrender.DirectionalLight(intensity=3.0), pose=np.eye(4))
    L2 = np.eye(4); L2[:3,3] = np.array([-1,-0.3,2]); scene.add(pyrender.DirectionalLight(intensity=1.8), pose=L2)

    renderer = pyrender.OffscreenRenderer(W, H)
    rgba, _ = renderer.render(scene)
    renderer.delete()

    # =============================
    # 3) EXACT HER CROP (NOW WORKS)
    # =============================
    uv = project_uv(verts_cam, fx, fy, cx, cy)
    umin, vmin = uv.min(axis=0)
    umax, vmax = uv.max(axis=0)
    bw, bh = umax - umin, vmax - vmin

    left   = max(0, int(umin - PAD_X*bw))
    right  = min(W, int(umax + PAD_X*bw))
    top    = max(0, int(vmin - PAD_Y*bh))
    bottom = min(H, int(vmin + UPPER_FRAC*bh))

    crop = rgba[top:bottom, left:right, :3]

    Image.fromarray(crop).save(f"{OUT_FRAMES}/{idx:05d}.png")

print("[DONE] Frames saved at:", OUT_FRAMES)
