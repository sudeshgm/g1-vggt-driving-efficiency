import json, sys, os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
W = json.load(open("manifests/windows.json"))["windows"]
w = next(w for w in W if w["state"] == sys.argv[1]) if len(sys.argv) > 1 else W[0]
f = next(f for f in w["frames"] if "depth_gt" in f)
z = np.load(f["depth_gt"]); img = Image.open("/workspace/data/" + f["file"])
plt.figure(figsize=(16, 9)); plt.imshow(img)
plt.scatter(z["u"], z["v"], c=z["d"].astype(float), s=1.5, cmap="turbo", vmin=0, vmax=80)
plt.colorbar(label="LiDAR depth (m)"); plt.title(f"{w['window_id']} frame {f['idx']} lidar_dt={f['lidar_dt_us']/1e3:.1f} ms")
os.makedirs("results/block1", exist_ok=True); p = f"results/block1/overlay_{w['window_id']}_f{f['idx']}.png"
plt.axis("off"); plt.savefig(p, dpi=80, bbox_inches="tight"); print(p)
