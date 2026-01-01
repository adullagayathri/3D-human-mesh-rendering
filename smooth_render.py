import os
import numpy as np
from scipy.signal import savgol_filter
from glob import glob
from scipy.spatial.transform import Rotation as R, Slerp

# -------------------------
# Paths
# -------------------------
NPZ_FOLDER = "/content/drive/MyDrive/Extracted_Parameters"
OUT_FOLDER_GV = "/content/drive/MyDrive/Filtered_GV"
OUT_FOLDER_SG = "/content/drive/MyDrive/Filtered_SG"

os.makedirs(OUT_FOLDER_GV, exist_ok=True)
os.makedirs(OUT_FOLDER_SG, exist_ok=True)

PERSON_ID = 0
MAX_STEP_DEG = 45.0  # maximum allowed rotation jump in degrees

# -------------------------
# Gaussian Volterra filter (temporal smoothing)
# -------------------------
def gv_filter(data, alpha=0.6):
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for t in range(1, len(data)):
        smoothed[t] = alpha * data[t] + (1-alpha) * smoothed[t-1]
    return smoothed

# -------------------------
# Savitzky-Golay filter
# -------------------------
def sg_filter(data, window=5, poly=2):
    if len(data) < window:
        window = len(data) if len(data)%2==1 else len(data)-1
    return savgol_filter(data, window_length=window, polyorder=poly, axis=0)

# -------------------------
# SLERP-based rotation stabilization (only for root)
# -------------------------
def stabilize_rotvecs(rotvecs, max_step_deg=45.0):
    N = len(rotvecs)
    rots = R.from_rotvec(rotvecs)
    quats = rots.as_quat()

    # Ensure shortest path
    for i in range(1, N):
        if np.dot(quats[i-1], quats[i]) < 0:
            quats[i] *= -1

    max_rad = np.deg2rad(max_step_deg)
    for i in range(1, N):
        r_prev = R.from_quat(quats[i-1])
        r_curr = R.from_quat(quats[i])
        r_delta = r_prev.inv() * r_curr
        angle = np.linalg.norm(r_delta.as_rotvec())
        if angle > max_rad:
            factor = max_rad / angle
            new_rotvec = r_prev.as_rotvec() + r_delta.as_rotvec() * factor
            quats[i] = R.from_rotvec(new_rotvec).as_quat()

    times = np.arange(N)
    slerp = Slerp(times, R.from_quat(quats))
    return slerp(times).as_rotvec()

# -------------------------
# Load all NPZ frames
# -------------------------
files = sorted(glob(os.path.join(NPZ_FOLDER, "*.npz")))
print(f"[INFO] Total frames: {len(files)}")

root_poses = []
body_poses = []
lhand_poses = []
rhand_poses = []
jaw_poses = []
shapes = []
exprs = []

for fname in files:
    Z = np.load(fname, allow_pickle=False)
    p = f"person_{PERSON_ID}_smplx_"
    root_poses.append(Z[p+"root_pose"].reshape(1,-1))
    body_poses.append(Z[p+"body_pose"].reshape(1,-1))
    lhand_poses.append(Z[p+"lhand_pose"].reshape(1,-1))
    rhand_poses.append(Z[p+"rhand_pose"].reshape(1,-1))
    jaw_poses.append(Z[p+"jaw_pose"].reshape(1,-1))
    shapes.append(Z[p+"shape"].reshape(1,-1))
    exprs.append(Z[p+"expr"].reshape(1,-1))

root_poses = np.vstack(root_poses)
body_poses = np.vstack(body_poses)
lhand_poses = np.vstack(lhand_poses)
rhand_poses = np.vstack(rhand_poses)
jaw_poses = np.vstack(jaw_poses)
shapes = np.vstack(shapes)
exprs = np.vstack(exprs)

# -------------------------
# Apply temporal smoothing
# -------------------------
root_gv   = gv_filter(root_poses)
body_gv   = gv_filter(body_poses)
lhand_gv  = gv_filter(lhand_poses)
rhand_gv  = gv_filter(rhand_poses)
jaw_gv    = gv_filter(jaw_poses)

root_sg   = sg_filter(root_poses)
body_sg   = sg_filter(body_poses)
lhand_sg  = sg_filter(lhand_poses)
rhand_sg  = sg_filter(rhand_poses)
jaw_sg    = sg_filter(jaw_poses)

# -------------------------
# Stabilize root rotations using SLERP
# -------------------------
root_gv   = stabilize_rotvecs(root_gv, MAX_STEP_DEG)
root_sg   = stabilize_rotvecs(root_sg, MAX_STEP_DEG)

# -------------------------
# Save filtered & stabilized NPZ
# -------------------------
for idx, fname in enumerate(files):
    Z = np.load(fname, allow_pickle=False)
    p = f"person_{PERSON_ID}_smplx_"

    # GV
    out_gv = os.path.join(OUT_FOLDER_GV, os.path.basename(fname))
    np.savez_compressed(out_gv,
        **{k: Z[k] for k in Z.keys() if not k.startswith(p)},
        **{
            p+"root_pose": root_gv[idx],
            p+"body_pose": body_gv[idx],
            p+"lhand_pose": lhand_gv[idx],
            p+"rhand_pose": rhand_gv[idx],
            p+"jaw_pose": jaw_gv[idx],
            p+"shape": shapes[idx],
            p+"expr": exprs[idx],
        }
    )

    # SG
    out_sg = os.path.join(OUT_FOLDER_SG, os.path.basename(fname))
    np.savez_compressed(out_sg,
        **{k: Z[k] for k in Z.keys() if not k.startswith(p)},
        **{
            p+"root_pose": root_sg[idx],
            p+"body_pose": body_sg[idx],
            p+"lhand_pose": lhand_sg[idx],
            p+"rhand_pose": rhand_sg[idx],
            p+"jaw_pose": jaw_sg[idx],
            p+"shape": shapes[idx],
            p+"expr": exprs[idx],
        }
    )

print("[DONE] Filtered and stabilized NPZs saved in two folders.")
