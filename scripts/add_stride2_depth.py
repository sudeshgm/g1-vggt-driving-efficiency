"""EXTRA CHECK (not stop rule): manifest with eval_idx = 0,2,...,30 and LiDAR GT depth for the odd-of-4 frames
(idx%4==2), same procedure as build_windows.py (nearest LIDAR_TOP within 30 ms, ego-motion compensated).
Writes manifests/windows_stride2.json; depth into bundle/depth/."""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from g1.nusc import NuMini
from g1.geometry import lidar_to_cam_T, load_lidar_bin, transform, sparse_depth_map
nu = NuMini("/workspace/data"); M = json.load(open("manifests/windows.json")); MAX = M["max_lidar_dt_us"]
lid = {sc["name"]: [r for r in nu.stream(sc["token"], "LIDAR_TOP") if os.path.exists(nu.path(r))] for sc in nu.scene}
cams = {r["token"]: r for sc in nu.scene for r in nu.stream(sc["token"], "CAM_FRONT")}
new = 0
for w in M["windows"]:
    for f in w["frames"]:
        if f["idx"] % 4 != 2: continue
        c = cams[f["token"]]; l, dt = nu.nearest(lid[w["scene"]], c["timestamp"]); f["lidar_token"], f["lidar_dt_us"] = l["token"], dt
        if dt <= MAX:
            p = f"bundle/depth/{c['token']}.npz"
            if not os.path.exists(p):
                Tlc = lidar_to_cam_T(nu.cs[l["calibrated_sensor_token"]], nu.ego_pose(l), nu.cam_cs(c), nu.ego_pose(c))
                D = sparse_depth_map(transform(Tlc, load_lidar_bin(nu.path(l))), nu.K(c), (900, 1600)); ys, xs = np.nonzero(D)
                np.savez_compressed(p, v=ys.astype(np.int16), u=xs.astype(np.int16), d=D[ys, xs].astype(np.float16)); new += 1
            f["depth_gt"] = p
    w["eval_idx"] = list(range(0, 32, 2)); w["eval_with_depth"] = [f["idx"] for f in w["frames"] if f["idx"] % 2 == 0 and "depth_gt" in f]
M["eval_stride"] = 2; M["note"] = "extra check only; pre-registered scoring is stride 4"
json.dump(M, open("manifests/windows_stride2.json", "w"), indent=1)
n = sum(len(w["eval_with_depth"]) for w in M["windows"]); print("new depth files", new, "; stride-2 frames with depth", n, "/", 16 * len(M["windows"]))
