"""Build non-overlapping 32-frame CAM_FRONT 12 Hz windows, ego-motion labels, and sparse LiDAR GT depth
for the evaluation-subset frames (idx % 4 == 0), using the nearest LIDAR_TOP sample_data (keyframe or sweep)
with ego-motion compensation. Outputs manifests/windows.json, manifests/frames_*.txt, data bundle under bundle/."""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from g1.nusc import NuMini
from g1.geometry import lidar_to_cam_T, load_lidar_bin, transform, sparse_depth_map
from g1.egomotion import ego_stats, label
from g1.metrics import evaluation_subset

ROOT = "/workspace/data"; N = 32; MAX_DT_US = 30000
out = sys.argv[1] if len(sys.argv) > 1 else "bundle"
nu = NuMini(ROOT); os.makedirs(f"{out}/depth", exist_ok=True); os.makedirs("manifests", exist_ok=True)
wins, skipped = [], []
for sc in nu.scene:
    cams = [r for r in nu.stream(sc["token"], "CAM_FRONT") if os.path.exists(nu.path(r))]
    lid = [r for r in nu.stream(sc["token"], "LIDAR_TOP") if os.path.exists(nu.path(r))]
    ts = np.array([c["timestamp"] for c in cams]); gaps = np.diff(ts) * 1e-6
    for w in range(len(cams) // N):
        fr = cams[w * N:(w + 1) * N]
        g = np.diff([c["timestamp"] for c in fr]) * 1e-6
        if g.max() > 0.2: skipped.append((sc["name"], w, "frame gap %.2fs" % g.max())); continue
        st = ego_stats([nu.ego_pose(c) for c in fr])
        frames = []
        for i, c in enumerate(fr):
            T = nu.cam_to_world(c)
            f = dict(idx=i, token=c["token"], file=c["filename"], timestamp=c["timestamp"], is_key_frame=c["is_key_frame"],
                     K=nu.K(c).tolist(), cam_to_world=T.tolist())
            if i in evaluation_subset(N, 4):
                l, dt = nu.nearest(lid, c["timestamp"])
                f["lidar_token"], f["lidar_dt_us"] = l["token"], dt
                if dt <= MAX_DT_US:
                    Tlc = lidar_to_cam_T(nu.cs[l["calibrated_sensor_token"]], nu.ego_pose(l), nu.cam_cs(c), nu.ego_pose(c))
                    D = sparse_depth_map(transform(Tlc, load_lidar_bin(nu.path(l))), nu.K(c), (900, 1600))
                    p = f"{out}/depth/{c['token']}.npz"; ys, xs = np.nonzero(D)
                    np.savez_compressed(p, v=ys.astype(np.int16), u=xs.astype(np.int16), d=D[ys, xs].astype(np.float16))
                    f["depth_gt"] = p; f["n_lidar_px"] = int(len(ys))
            frames.append(f)
        wid = f"{sc['name']}_w{w}"
        wins.append(dict(window_id=wid, scene=sc["name"], description=sc["description"], n=N, ego=st, state=label(st),
                         frames=frames, eval_idx=evaluation_subset(N, 4).tolist(),
                         eval_with_depth=[f["idx"] for f in frames if "depth_gt" in f]))
json.dump(dict(windows=wins, skipped=skipped, n_frames=N, eval_stride=4, max_lidar_dt_us=MAX_DT_US),
          open("manifests/windows.json", "w"), indent=1)
for stride in (1, 2, 4):
    with open(f"manifests/frames_stride{stride}.txt", "w") as fh:
        for w in wins:
            fh.write(w["window_id"] + " " + " ".join(f["file"] for f in w["frames"][::stride]) + "\n")
from collections import Counter
print(len(wins), "windows;", Counter(w["state"] for w in wins), "skipped", skipped)
