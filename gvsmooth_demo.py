#!/usr/bin/env python3
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import argparse
import numpy as np
import torch
import smplx
import pyrender
import trimesh
from PIL import Image

# -------------------------
# ARGPARSE
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser("Render SMPL-X mesh frames")
    parser.add_argument("--npz_folder", type=str, required=True, help="Folder with filtered NPZ params")
    parser.add_argument("--output_frames", type=str, required=True, help="Folder to save rendered PNG frames")
    parser.add_argument("--model_path", type=str, default="/content/drive/MyDrive/OSXBackup/OSX/common/utils/human_model_files/smplx/SMPLX_NEUTRAL.npz", help="Path to SMPL-X model")
    parser.add_argument("--person_id", type=int, default=0, help="Person ID (default: 0)")
    return parser.parse_args()

# -------------------------
# UV Projection
# -------------------------
def project_uv(verts_cam, fx, fy, cx, cy):
    X, Y, Z = verts_cam[:,0], verts_cam[:,1], verts_cam[:,2]
    u = fx * (X/Z) + cx
    v = fy * (Y/Z) + cy
    return np.stack([u,v], axis=1)

# -------------------------
# MAIN
# -------------------------
def main():
    args = parse_args()
    NPZ_FOLDER = args.npz_folder
    OUTPUT_DIR = args.output_frames
    PID = args.person_id

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    H, W = 720, 1280
    PAD_X, PAD_Y = 0.50, 0.14
    UPPER_FRAC = 0.55
    MARGIN = 0.05
    BG = (245, 245, 245, 255)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load SMPL-X model
    model = smplx.create(
        model_path=args.model_path,
        model_type="smplx",
        gender="neutral",
        num_betas=10,
        num_expression_coeffs=10,
        use_pca=False,
        use_face_contour=True,
        ext="npz"
    ).to(device).eval()

    files = sorted([f for f in os.listdir(NPZ_FOLDER) if f.endswith(".npz")])
    print(f"[INFO] Total NPZ frames: {len(files)}")

    frame_count = 0
    for idx, fname in enumerate(files):
        Z = np.load(os.path.join(NPZ_FOLDER, fname), allow_pickle=True)
        p = f"person_{PID}_smplx_"

        params = {
            "global_orient":  Z[p+"root_pose"].reshape(1,-1),
            "body_pose":      Z[p+"body_pose"].reshape(1,-1),
            "left_hand_pose": Z[p+"lhand_pose"].reshape(1,-1),
            "right_hand_pose":Z[p+"rhand_pose"].reshape(1,-1),
            "jaw_pose":       Z[p+"jaw_pose"].reshape(1,3),
            "betas":          Z[p+"shape"].reshape(1,-1),
            "expression":     Z[p+"expr"].reshape(1,-1),
            "leye_pose":      np.zeros((1,3), dtype=np.float32),
            "reye_pose":      np.zeros((1,3), dtype=np.float32)
        }

        cam_trans = Z.get(f"person_{PID}_cam_trans", np.zeros(3, dtype=np.float32))
        fx, fy = Z.get(f"person_{PID}_focal", [5000.0, 5000.0])
        cx, cy = Z.get(f"person_{PID}_princpt", [640.0, 360.0])

        # Build mesh
        with torch.no_grad():
            tens = {k: torch.tensor(v, dtype=torch.float32).to(device) for k,v in params.items()}
            verts = model(**tens).vertices[0].cpu().numpy()
        if np.isnan(verts).any():
            print(f"[WARN] Skipping frame {idx} due to NaN vertices")
            continue
        faces = model.faces
        verts_cam = verts + cam_trans[None,:]

        # Auto-fit to viewport
        uv = project_uv(verts_cam, fx, fy, cx, cy)
        umin, vmin = uv.min(axis=0); umax, vmax = uv.max(axis=0)
        bbox_w, bbox_h = umax-umin, vmax-vmin
        W_eff, H_eff = W*(1.0-2*MARGIN), H*(1.0-2*MARGIN)
        k = max(bbox_w/max(W_eff,1e-6), bbox_h/max(H_eff,1e-6), 1.0)
        verts_cam[:,2] *= k
        uv = project_uv(verts_cam, fx, fy, cx, cy)
        uc, vc = uv.mean(axis=0)
        du, dv = (W/2.0 - uc, H/2.0 - vc)
        z_mean = np.median(verts_cam[:,2])
        verts_cam[:,0] += (du * z_mean)/fx
        verts_cam[:,1] += (dv * z_mean)/fy

        verts_gl = verts_cam.copy()
        verts_gl[:,1] *= -1
        verts_gl[:,2] *= -1

        # Render
        scene = pyrender.Scene(bg_color=[c/255.0 for c in BG])
        material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.86,0.86,0.86,1.0], metallicFactor=0.0, roughnessFactor=0.7)
        tri = trimesh.Trimesh(verts_gl, faces, process=False)
        mesh = pyrender.Mesh.from_trimesh(tri, material=material, smooth=True)
        scene.add(mesh)
        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
        scene.add(camera, pose=np.eye(4, dtype=np.float32))
        scene.add(pyrender.DirectionalLight(intensity=3.0), pose=np.eye(4, dtype=np.float32))
        L2 = np.eye(4, dtype=np.float32); L2[:3,3] = np.array([-1.0,-0.3,2.0], dtype=np.float32)
        scene.add(pyrender.DirectionalLight(intensity=1.8), pose=L2)
        renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
        rgba,_ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        renderer.delete()

        # Body-aware crop (upper torso)
        uv = project_uv(verts_cam, fx, fy, cx, cy)
        umin, vmin = uv.min(axis=0); umax, vmax = uv.max(axis=0)
        bbox_w, bbox_h = umax-umin, vmax-vmin
        left = max(0.0, umin-PAD_X*bbox_w)
        right= min(float(W), umax+PAD_X*bbox_w)
        top  = max(0.0, vmin-PAD_Y*bbox_h)
        bottom = min(float(H), vmin+UPPER_FRAC*bbox_h)
        x0,y0,x1,y1 = int(round(left)),int(round(top)),int(round(right)),int(round(bottom))
        if x1<=x0 or y1<=y0: x0,y0,x1,y1=0,0,W,H
        crop = rgba[y0:y1, x0:x1, :3]

        out_path = os.path.join(OUTPUT_DIR, f"{frame_count:05d}.png")
        Image.fromarray(crop).save(out_path)
        frame_count += 1
        print(f"[INFO] Saved frame {frame_count}")

if __name__ == "__main__":
    main()
